import asyncio
import functools
import logging
from dataclasses import dataclass

import chess
import chess.engine
import chess.polyglot
import uvloop
from tqdm import tqdm

from dfrc_analysis.analysis.config import AnalysisConfig, load_config
from dfrc_analysis.analysis.results import (
    PositionAnalysis,
    PositionNode,
)
from dfrc_analysis.positions.positions import get_chess960_position
from dfrc_analysis.utils import calculate_subtree_size

AnalysisTree = PositionNode
logger = logging.getLogger(__name__)

# Decorate the imported utility with LRU cache.
calculate_subtree_size = functools.lru_cache(maxsize=2048)(calculate_subtree_size)


def count_nodes(node: PositionNode) -> int:
    """Recursively counts the total number of nodes in a PositionNode tree."""
    count = 1  # Count the current node
    if node.children:
        for child in node.children:
            count += count_nodes(child)
    return count


class AsyncTranspositionTable:
    """
    Async-aware cache using Zobrist Hashing (int) for high-performance lookups.
    Prevents double-work by sharing Futures.
    """

    def __init__(self) -> None:
        self._cache: dict[int, asyncio.Future[PositionNode]] = {}

    def get_future(self, key: int) -> tuple[asyncio.Future[PositionNode], bool]:
        """
        Returns (Future, is_new).
        """
        if key in self._cache:
            return self._cache[key], False

        future = asyncio.Future()
        self._cache[key] = future
        return future, True


class AsyncEngineManager:
    """
    Manages the lifecycle of async chess engines.
    """

    def __init__(self, engine_path: str, options: dict, max_engines: int) -> None:
        self.engine_path = engine_path
        self.options = options
        self.semaphore = asyncio.Semaphore(max_engines)
        self._engines_stack: list[chess.engine.Protocol] = []

    async def get_engine(self) -> chess.engine.Protocol:
        await self.semaphore.acquire()
        if self._engines_stack:
            return self._engines_stack.pop()

        _, engine = await chess.engine.popen_uci(self.engine_path)
        await engine.configure(self.options)
        return engine

    async def return_engine(self, engine: chess.engine.Protocol) -> None:
        self._engines_stack.append(engine)
        self.semaphore.release()

    async def cleanup(self) -> None:
        if self._engines_stack:
            await asyncio.gather(*(engine.quit() for engine in self._engines_stack))


@dataclass
class AsyncRecursiveAnalyzer:
    root_boards: list[chess.Board]
    engine_manager: AsyncEngineManager
    cfg: AnalysisConfig
    tt: AsyncTranspositionTable
    only_terminal_pv: bool = True

    @functools.cached_property
    def max_positions(self) -> int:
        single_tree_size = calculate_subtree_size(
            0,
            self.cfg.analysis_depth_ply,
            tuple(self.cfg.num_top_moves_per_ply),
        )
        return single_tree_size * len(self.root_boards)

    @functools.cached_property
    def progress_bar(self) -> tqdm:
        return tqdm(
            total=self.max_positions,
            desc=f"Analyzing {len(self.root_boards)} positions",
            unit="node",
        )

    def _update_pbar(self, amount: int) -> None:
        self.progress_bar.update(amount)

    def _extract_eval(
        self,
        score: chess.engine.PovScore,
    ) -> tuple[int | None, int | None]:
        """Helper to extract (centipawns, mate) from a score object relative to White."""
        cp_val = score.white().score()
        return (cp_val, None) if cp_val is not None else (None, score.white().mate())

    async def _get_candidates(
        self,
        engine: chess.engine.Protocol,
        board: chess.Board,
        ply: int,
    ) -> list[chess.engine.InfoDict]:
        return await engine.analyse(
            board,
            chess.engine.Limit(depth=self.cfg.stockfish_depth_per_ply[ply]),
            multipv=self.cfg.num_top_moves_per_ply[ply],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )

    async def _analyze_node(
        self,
        board: chess.Board,
        ply: int,
        tree_index: int,
        incoming_move: str = "root",
        incoming_analysis: PositionAnalysis | None = None,
    ) -> PositionNode | None:
        """
        Main recursive entry point.
        """
        # --- 1. Leaf Handling ---
        if ply >= self.cfg.analysis_depth_ply:
            return PositionNode(
                move=incoming_move,
                children=[],
                analysis=incoming_analysis
                or PositionAnalysis(cpl=None, mate=None, pv=None),
            )

        # --- 2. Cache Lookup ---
        board_hash = chess.polyglot.zobrist_hash(board)
        future, is_new = self.tt.get_future(board_hash)

        if not is_new:
            cached_node = await future
            if cached_node:
                skipped_work = calculate_subtree_size(
                    ply,
                    self.cfg.analysis_depth_ply,
                    tuple(self.cfg.num_top_moves_per_ply),
                )
                self._update_pbar(skipped_work)
                return cached_node

        # --- 3. Compute Node ---
        try:
            result = await self._compute_node(board, ply, tree_index, incoming_move)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise e

    async def _compute_node(
        self,
        board: chess.Board,
        ply: int,
        tree_index: int,
        node_move: str,
    ) -> PositionNode | None:
        """
        Runs the engine, determines children, and recursively calls _analyze_node.
        """
        # Run Engine
        engine = await self.engine_manager.get_engine()
        try:
            candidates = await self._get_candidates(engine, board, ply)
            self._update_pbar(1)
        finally:
            await self.engine_manager.return_engine(engine)

        if not candidates:
            return None

        best_candidate = candidates[0]

        if (score := best_candidate.get("score")) is None:
            raise RuntimeError(f"Failed to get score at ply {ply}")

        cpl_val, mate_val = self._extract_eval(score)
        pv_moves = [m.uci() for m in best_candidate.get("pv", [])]

        # --- Logging ---
        # Format score for display
        score_str = f"M{mate_val}" if mate_val is not None else f"{cpl_val:+d}"
        best_move_uci = pv_moves[0] if pv_moves else "none"

        # Send info to tqdm
        self.progress_bar.write(
            f"Tree #{tree_index} | Ply {ply} | {node_move:<5} -> {best_move_uci:<5} ({score_str})",
        )

        # --- Terminal Logic ---
        is_terminal = mate_val is not None or (
            cpl_val is not None and abs(cpl_val) >= self.cfg.balanced_threshold
        )

        if is_terminal and ply < self.cfg.analysis_depth_ply:
            subtree_size = calculate_subtree_size(
                ply + 1,
                self.cfg.analysis_depth_ply,
                tuple(self.cfg.num_top_moves_per_ply),
            )
            pruned_nodes = len(candidates) * subtree_size
            if pruned_nodes > 0:
                self._update_pbar(pruned_nodes)

        children = []

        # --- Recursive Expansion ---
        if not is_terminal:
            child_tasks = []
            for candidate in candidates:
                cand_pv = candidate.get("pv", [])
                if not cand_pv:
                    continue

                move_obj = cand_pv[0]
                move_uci = move_obj.uci()

                child_cpl, child_mate = None, None
                if child_score := candidate.get("score"):
                    child_cpl, child_mate = self._extract_eval(child_score)

                child_analysis = PositionAnalysis(
                    cpl=child_cpl,
                    mate=child_mate,
                    pv=None,
                )

                new_board = board.copy(stack=False)
                new_board.push(move_obj)

                child_tasks.append(
                    self._analyze_node(
                        new_board,
                        ply + 1,
                        tree_index=tree_index,  # Pass identity down
                        incoming_move=move_uci,
                        incoming_analysis=child_analysis,
                    ),
                )

            results = await asyncio.gather(*child_tasks)
            children = [r for r in results if r is not None]

        return PositionNode(
            move=node_move,
            children=children,
            analysis=PositionAnalysis(cpl=cpl_val, mate=mate_val, pv=pv_moves),
        )

    async def analyse(self) -> list[AnalysisTree]:
        self._update_pbar(0)
        try:
            root_tasks = []
            # Pass the index (i) as the tree identifier
            for i, board in enumerate(self.root_boards):
                task = self._analyze_node(
                    board.copy(stack=False),
                    0,
                    tree_index=i,
                    incoming_move="root",
                    incoming_analysis=None,
                )
                root_tasks.append(task)

            results = await asyncio.gather(*root_tasks)
            return [r for r in results if r is not None]
        finally:
            self.progress_bar.close()


def analyse_dfrc_batch(
    positions: list[tuple[int, int]],
    cfg_id: str,
    threads: int,
    hash_size: int,
    engine_path: str = "stockfish",
    verbose: bool = False,
) -> list[AnalysisTree]:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    chess_engine_logger = logging.getLogger("chess.engine")
    chess_engine_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    boards = []
    for white_id, black_id in positions:
        w_fen = get_chess960_position(white_id)
        b_fen = get_chess960_position(black_id)
        board = chess.Board(chess960=True)
        board.set_fen(
            f"{b_fen.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{w_fen.upper()} w - - 0 1",
        )
        boards.append(board)

    logger.info(f"Preparing batch analysis for {len(boards)} positions...")

    cfg = load_config(cfg_id)
    options = {"Threads": 1, "Hash": hash_size}

    async def run_async_analysis() -> list[AnalysisTree]:
        tt = AsyncTranspositionTable()
        engine_manager = AsyncEngineManager(
            engine_path,
            options,
            max_engines=threads,
        )

        analyzer = AsyncRecursiveAnalyzer(
            root_boards=boards,
            engine_manager=engine_manager,
            cfg=cfg,
            tt=tt,
        )

        try:
            return await analyzer.analyse()
        finally:
            await engine_manager.cleanup()

    return asyncio.run(run_async_analysis())


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)

    # Configuration
    BATCH_SIZE = 8
    batch_positions = [(i, i) for i in range(BATCH_SIZE)]

    CFG_ID = "XS"
    THREADS = 8
    HASH_SIZE = 256

    logger.info("--- Starting Batch Analysis ---")
    start_time = time.perf_counter()

    trees = analyse_dfrc_batch(
        positions=batch_positions,
        cfg_id=CFG_ID,
        threads=THREADS,
        hash_size=HASH_SIZE,
        verbose=False,
    )

    end_time = time.perf_counter()
    duration = end_time - start_time

    logger.info("-------------------------")
    logger.info(f"Batch complete. Processed {len(trees)} trees in {duration:.4f}s")
    logger.info("-------------------------")

    for i, tree in enumerate(trees):
        node_count = count_nodes(tree)
        root_children = []
        if tree.children:
            root_children = [child.move for child in tree.children]

        logger.info(
            f"Tree #{i:<2} | Nodes: {node_count:<5} | Root Children: {root_children}",
        )

    logger.info("-------------------------")
