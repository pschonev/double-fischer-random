import asyncio
import functools
import logging
from dataclasses import dataclass, field

import chess
import chess.engine
import chess.polyglot
import uvloop
from tqdm import tqdm

from dfrc_analysis.analysis.config import AnalysisConfig, load_config
from dfrc_analysis.analysis.results import (
    AnalysisParams,
    PositionAnalysis,
    PositionNode,
)
from dfrc_analysis.positions.positions import get_chess960_position
from dfrc_analysis.utils import calculate_subtree_size

AnalysisTree = PositionNode
logger = logging.getLogger(__name__)

# Decorate the imported utility with LRU cache.
calculate_subtree_size = functools.lru_cache(maxsize=2048)(calculate_subtree_size)


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
    root_board: chess.Board
    engine_manager: AsyncEngineManager
    cfg: AnalysisConfig
    tt: AsyncTranspositionTable
    only_terminal_pv: bool = True

    @functools.cached_property
    def max_positions(self) -> int:
        return calculate_subtree_size(
            0,
            self.cfg.analysis_depth_ply,
            tuple(self.cfg.num_top_moves_per_ply),
        )

    @functools.cached_property
    def progress_bar(self) -> tqdm:
        return tqdm(total=self.max_positions, desc="Analyzing positions")

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
        incoming_move: str = "root",
        incoming_analysis: PositionAnalysis | None = None,
    ) -> PositionNode | None:
        """
        Main recursive entry point.
        1. Checks recursion depth (Leaf logic).
        2. Checks Cache (Transposition Table).
        3. Computes Node (Engine analysis).
        """

        # --- 1. Leaf Handling ---
        # If we reached the depth limit, we stop expansion.
        # We return a node based on the data passed down from the parent.
        if ply >= self.cfg.analysis_depth_ply:
            return PositionNode(
                move=incoming_move,
                children=[],
                # Use the analysis passed from parent (e.g. static eval from previous ply)
                analysis=incoming_analysis
                or PositionAnalysis(cpl=None, mate=None, pv=None),
            )

        # --- 2. Cache Lookup ---
        board_hash = chess.polyglot.zobrist_hash(board)
        # We include ply in cache key conceptualization implicitly, but strictly
        # a position is a position. However, if we hit it at different depths,
        # we might want to be careful. For simplicity, we assume same-ply hits here.
        future, is_new = self.tt.get_future(board_hash)

        if not is_new:
            cached_node = await future
            if cached_node:
                # If we hit cache, we technically skip the computation of this subtree
                skipped_work = calculate_subtree_size(
                    ply,
                    self.cfg.analysis_depth_ply,
                    tuple(self.cfg.num_top_moves_per_ply),
                )
                self._update_pbar(skipped_work)

                # Return a copy/reference, but likely update the 'move' to match current path
                # if the user cares about path-consistency.
                # For now, returning the cached object is standard TT behavior.
                return cached_node

        # --- 3. Compute Node ---
        try:
            result = await self._compute_node(board, ply, incoming_move)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise e

    async def _compute_node(
        self, board: chess.Board, ply: int, node_move: str
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

        # Best move logic (for analysis display of THIS node)
        best_candidate = candidates[0]

        # Safety check for score
        if (score := best_candidate.get("score")) is None:
            raise RuntimeError(f"Failed to get score at ply {ply}")

        cpl_val, mate_val = self._extract_eval(score)

        # Determine PV for this node
        pv_moves = [m.uci() for m in best_candidate.get("pv", [])]

        # --- Terminal Logic ---
        # Check if we should stop expanding due to game-over or eval threshold
        is_terminal = mate_val is not None or (
            cpl_val is not None and abs(cpl_val) >= self.cfg.balanced_threshold
        )

        # If we are pruning early (before max depth), account for skipped nodes in progress bar
        if is_terminal and ply < self.cfg.analysis_depth_ply:
            subtree_size = calculate_subtree_size(
                ply + 1,
                self.cfg.analysis_depth_ply,
                tuple(self.cfg.num_top_moves_per_ply),
            )
            # We prune all candidates' branches
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

                # Extract score for the child to pass down
                child_cpl, child_mate = None, None
                if child_score := candidate.get("score"):
                    child_cpl, child_mate = self._extract_eval(child_score)

                child_analysis = PositionAnalysis(
                    cpl=child_cpl, mate=child_mate, pv=None
                )

                # Create next board state
                new_board = board.copy(stack=False)
                new_board.push(move_obj)

                # Recurse: Pass the move and the analysis we just found
                child_tasks.append(
                    self._analyze_node(
                        new_board,
                        ply + 1,
                        incoming_move=move_uci,
                        incoming_analysis=child_analysis,
                    )
                )

            # Gather results
            results = await asyncio.gather(*child_tasks)
            children = [r for r in results if r is not None]

        # Construct final node
        # Note: We use node_move (passed in) as the identifier for this node
        return PositionNode(
            move=node_move,
            children=children,
            analysis=PositionAnalysis(cpl=cpl_val, mate=mate_val, pv=pv_moves),
        )

    async def analyse(self) -> AnalysisTree:
        try:
            # Root call
            return await self._analyze_node(
                self.root_board.copy(stack=False),
                0,
                incoming_move="root",
                incoming_analysis=None,
            )
        finally:
            self.progress_bar.close()


def analyse_dfrc_position(
    params: AnalysisParams,
    engine_path: str = "stockfish",
    *,
    verbose: bool = False,
) -> AnalysisTree:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    chess_engine_logger = logging.getLogger("chess.engine")
    chess_engine_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    white, black = (
        get_chess960_position(params.white_id),
        get_chess960_position(params.black_id),
    )

    board = chess.Board(chess960=True)
    board.set_fen(
        f"{black.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{white.upper()} w - - 0 1",
    )
    logger.info(
        f"Analyzing position: {params.white_id=} {params.black_id=}\n{board.fen()}",
    )

    cfg = load_config(params.cfg_id)
    options = {"Threads": 1, "Hash": params.hash}

    async def run_async_analysis() -> AnalysisTree:
        tt = AsyncTranspositionTable()
        engine_manager = AsyncEngineManager(
            engine_path,
            options,
            max_engines=params.threads,
        )

        analyzer = AsyncRecursiveAnalyzer(
            root_board=board,
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

    # Setup parameters
    params = AnalysisParams(
        white_id=0,
        black_id=0,
        cfg_id="XS",
        threads=8,
        hash=4096,
    )

    logger.info("--- Starting User Script Analysis ---")
    start_time = time.perf_counter()

    # Run analysis
    tree = analyse_dfrc_position(params=params, verbose=False)

    end_time = time.perf_counter()
    duration = end_time - start_time

    logger.info(f"""
          -------------------------
          Analysis tree:
            {tree}
          """)
    logger.info("-------------------------")
    logger.info(f"Execution Time: {duration:.4f} seconds")
    logger.info("-------------------------")
