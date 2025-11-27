import asyncio
import functools
import logging
from dataclasses import dataclass

import chess
import chess.engine
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


@dataclass
class RecursiveAnalyzer:
    root_board: chess.Board
    engine: chess.engine.Protocol
    cfg: AnalysisConfig
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
        board: chess.Board,
        ply: int,
    ) -> list[chess.engine.InfoDict]:
        # Send command to the single shared engine.
        return await self.engine.analyse(
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
        """
        # --- 1. Leaf Handling ---
        if ply >= self.cfg.analysis_depth_ply:
            return PositionNode(
                move=incoming_move,
                children=[],
                analysis=incoming_analysis
                or PositionAnalysis(cpl=None, mate=None, pv=None),
            )

        # --- 2. Compute Node (Sequential) ---
        candidates = await self._get_candidates(board, ply)
        self._update_pbar(1)

        if not candidates:
            return None

        best_candidate = candidates[0]

        if (score := best_candidate.get("score")) is None:
            raise RuntimeError(f"Failed to get score at ply {ply}")

        cpl_val, mate_val = self._extract_eval(score)
        pv_moves = [m.uci() for m in best_candidate.get("pv", [])]

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

        # --- Recursive Expansion (Sequential Loop) ---
        if not is_terminal:
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

                # Await strictly sequentially to avoid race conditions
                child_node = await self._analyze_node(
                    new_board,
                    ply + 1,
                    incoming_move=move_uci,
                    incoming_analysis=child_analysis,
                )

                if child_node is not None:
                    children.append(child_node)

        return PositionNode(
            move=incoming_move,
            children=children,
            analysis=PositionAnalysis(cpl=cpl_val, mate=mate_val, pv=pv_moves),
        )

    async def analyse(self) -> AnalysisTree:
        try:
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
    # Standard asyncio event loop (no uvloop)

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

    # Configure ONE engine with ALL threads
    options = {"Threads": params.threads, "Hash": params.hash}

    async def run_analysis() -> AnalysisTree:
        # Create the single persistent engine
        _, engine = await chess.engine.popen_uci(engine_path)
        await engine.configure(options)

        analyzer = RecursiveAnalyzer(
            root_board=board,
            engine=engine,
            cfg=cfg,
        )

        try:
            return await analyzer.analyse()
        finally:
            await engine.quit()

    return asyncio.run(run_analysis())


if __name__ == "__main__":
    import time

    # Setup parameters
    params = AnalysisParams(
        white_id=0,
        black_id=0,
        cfg_id="XS",
        threads=8,  # 8 threads for the single engine
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
