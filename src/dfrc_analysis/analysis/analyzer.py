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
# Note: Callers must now pass hashable arguments (e.g., tuples instead of lists).
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
        Key is now a 64-bit integer (Zobrist Hash).
        """
        if key in self._cache:
            return self._cache[key], False

        # Create a new future bound to the current loop
        future = asyncio.Future()
        self._cache[key] = future
        return future, True


class AsyncEngineManager:
    """
    Manages the lifecycle of async chess engines.
    Uses a Semaphore to ensure we never exceed N concurrent engines.
    """

    def __init__(self, engine_path: str, options: dict, max_engines: int) -> None:
        self.engine_path = engine_path
        self.options = options
        self.semaphore = asyncio.Semaphore(max_engines)
        self._engines_stack: list[chess.engine.Protocol] = []

    async def get_engine(self) -> chess.engine.Protocol:
        """
        Acquires a permit (Semaphore) and provides an engine.
        If an engine is idle in the stack, reuse it.
        Otherwise, spawn a new one (up to max_engines).
        """
        await self.semaphore.acquire()

        if self._engines_stack:
            return self._engines_stack.pop()

        # Spawn a new engine process
        _, engine = await chess.engine.popen_uci(self.engine_path)
        await engine.configure(self.options)
        return engine

    async def return_engine(self, engine: chess.engine.Protocol) -> None:
        """Returns engine to the stack and releases semaphore."""
        self._engines_stack.append(engine)
        self.semaphore.release()

    async def cleanup(self) -> None:
        """Quit all cached engines."""
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
        """
        Calculates the total tree size.
        Cached because it is computationally expensive and constant for this config.
        """
        return calculate_subtree_size(
            0,
            self.cfg.analysis_depth_ply,
            tuple(self.cfg.num_top_moves_per_ply),
        )

    @functools.cached_property
    def progress_bar(self) -> tqdm:
        """
        Lazy-loaded progress bar.
        Only appears in stdout when .analyse() first accesses it.
        """
        return tqdm(total=self.max_positions, desc="Analyzing positions")

    def _update_pbar(self, amount: int) -> None:
        # Tqdm is not async-aware, but updating from the main thread is safe.
        self.progress_bar.update(amount)

    def _compute_eval(
        self,
        score: chess.engine.PovScore,
    ) -> tuple[int | None, int | None]:
        cp_val = score.white().score()
        return (cp_val, None) if cp_val is not None else (None, score.white().mate())

    async def _get_candidates(
        self,
        engine: chess.engine.Protocol,
        board: chess.Board,
        ply: int,
    ) -> list[chess.engine.InfoDict]:
        """Async wrapper for engine analysis."""
        return await engine.analyse(
            board,
            chess.engine.Limit(depth=self.cfg.stockfish_depth_per_ply[ply]),
            multipv=self.cfg.num_top_moves_per_ply[ply],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )

    async def _analyze_node(self, board: chess.Board, ply: int) -> PositionNode | None:
        # OPTIMIZATION: Zobrist Hash (int) instead of FEN (str)
        board_hash = chess.polyglot.zobrist_hash(board)
        future, is_new = self.tt.get_future(board_hash)

        if not is_new:
            cached_node = await future
            if cached_node:
                # FIX: Convert list to tuple for cache lookup
                skipped_work = calculate_subtree_size(
                    ply,
                    self.cfg.analysis_depth_ply,
                    tuple(self.cfg.num_top_moves_per_ply),
                )
                self._update_pbar(skipped_work)
            return cached_node

        try:
            result = await self._compute_node(board, ply)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise e

    async def _compute_node(self, board: chess.Board, ply: int) -> PositionNode | None:
        if ply >= self.cfg.analysis_depth_ply:
            return None

        engine = await self.engine_manager.get_engine()
        try:
            candidates = await self._get_candidates(engine, board, ply)
            self._update_pbar(1)
        finally:
            await self.engine_manager.return_engine(engine)

        if not candidates:
            return None

        current_candidate = candidates[0]
        pv_moves = current_candidate.get("pv", [])
        if not pv_moves:
            return None

        current_move = pv_moves[0].uci() if ply > 0 else "root"

        if (score := current_candidate.get("score")) is None:
            raise RuntimeError("Failed to get score")
        cpl_val, mate_val = self._compute_eval(score)

        is_terminal = (
            ply >= self.cfg.analysis_depth_ply
            or mate_val is not None
            or (cpl_val is not None and abs(cpl_val) >= self.cfg.balanced_threshold)
        )

        if is_terminal and ply < self.cfg.analysis_depth_ply:
            # FIX: Convert list to tuple for cache lookup
            subtree_size = calculate_subtree_size(
                ply + 1,
                self.cfg.analysis_depth_ply,
                tuple(self.cfg.num_top_moves_per_ply),
            )
            pruned_nodes = len(candidates) * subtree_size
            if pruned_nodes > 0:
                self._update_pbar(pruned_nodes)

        children = []
        if not is_terminal:
            child_tasks = []
            for candidate in candidates:
                pv = candidate.get("pv", [])
                if not pv:
                    continue

                # OPTIMIZATION: Efficient Copying (stack=False)
                new_board = board.copy(stack=False)
                new_board.push(pv[0])
                child_tasks.append(self._analyze_node(new_board, ply + 1))

            results = await asyncio.gather(*child_tasks)
            children = [r for r in results if r is not None]

        pv = None
        if ply == 0 or not self.only_terminal_pv or is_terminal:
            pv = [move.uci() for move in current_candidate.get("pv", [])]

        return PositionNode(
            move=current_move,
            children=children,
            analysis=PositionAnalysis(cpl=cpl_val, mate=mate_val, pv=pv),
        )

    async def analyse(self) -> AnalysisTree:
        try:
            return await self._analyze_node(self.root_board.copy(stack=False), 0)
        finally:
            self.progress_bar.close()


def analyse_dfrc_position(
    params: AnalysisParams,
    engine_path: str = "stockfish",
    *,
    verbose: bool = False,
) -> AnalysisTree:
    # OPTIMIZATION: Enforce uvloop policy
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
        white_id=0,  # Standard Chess
        black_id=0,
        cfg_id="XS",  # Ensure this config matches the depth/width below!
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
