import logging
from dataclasses import dataclass, field

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


@dataclass
class RecursiveEngineAnalyzer:
    board: chess.Board
    engine: chess.engine.SimpleEngine
    cfg: AnalysisConfig
    only_terminal_pv: bool = True
    progress_bar: tqdm = field(init=False)

    def __post_init__(self) -> None:
        # Calculate maximum possible positions to analyze using the consolidated function
        max_positions = calculate_subtree_size(
            0,
            self.cfg.analysis_depth_ply,
            self.cfg.num_top_moves_per_ply,
        )
        # Initialize progress bar
        self.progress_bar = tqdm(total=max_positions, desc="Analyzing positions")

    def _get_candidates(
        self,
        board: chess.Board,
        ply: int,
    ) -> list[chess.engine.InfoDict]:
        """Get engine analysis candidates for the given board state."""
        return self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.cfg.stockfish_depth_per_ply[ply]),
            multipv=self.cfg.num_top_moves_per_ply[ply],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )

    def _compute_eval(
        self,
        score: chess.engine.PovScore,
    ) -> tuple[int | None, int | None]:
        """Convert a PovScore to (centipawns, mate_score) tuple from White's perspective."""
        cp_val = score.white().score()
        return (cp_val, None) if cp_val is not None else (None, score.white().mate())

    def _build_analysis_tree(self, board: chess.Board, ply: int) -> PositionNode | None:  # noqa: C901
        """Recursively build the analysis tree."""
        # Update progress bar for this position
        self.progress_bar.update(1)
        candidates = self._get_candidates(board, ply)

        # Handle no legal moves (checkmate or stalemate)
        if not candidates:
            return None

        # Process the current position
        current_candidate = candidates[0]
        pv_moves = current_candidate.get("pv", [])
        if not pv_moves:
            return None

        # For non-root positions, get the move that led here
        current_move = pv_moves[0].uci() if ply > 0 else "root"

        # Get the score from the engine analysis
        if (score := current_candidate.get("score")) is None:
            raise RuntimeError("Failed to get score from engine analysis")
        cpl_val, mate_val = self._compute_eval(score)

        # Check if this is a terminal position
        is_terminal = (
            ply >= self.cfg.analysis_depth_ply - 1
            or mate_val is not None
            or (cpl_val is not None and abs(cpl_val) >= self.cfg.balanced_threshold)
        )

        # If terminal but not at max depth, count pruned nodes
        if is_terminal and ply < self.cfg.analysis_depth_ply - 1:
            # Calculate the size of the pruned subtree starting from next ply
            # Multiply by number of candidates since each candidate would have its own subtree
            pruned_nodes = len(candidates) * calculate_subtree_size(
                ply + 1,
                self.cfg.analysis_depth_ply,
                self.cfg.num_top_moves_per_ply,
            )
            if pruned_nodes > 0:
                self.progress_bar.update(pruned_nodes)
                self.progress_bar.refresh()  # Force update

        # Build children if not terminal
        children = []
        if not is_terminal:
            for candidate in candidates:
                pv_moves = candidate.get("pv", [])
                if not pv_moves:
                    continue

                new_board = board.copy()
                new_board.push(pv_moves[0])
                child_node = self._build_analysis_tree(new_board, ply + 1)
                if child_node:
                    children.append(child_node)

        # Principal variation handling
        pv = None
        if ply == 0 or not self.only_terminal_pv or is_terminal:
            pv = [move.uci() for move in current_candidate.get("pv", [])]

        return PositionNode(
            move=current_move,
            children=children,
            analysis=PositionAnalysis(cpl=cpl_val, mate=mate_val, pv=pv),
        )

    def analyse(
        self,
    ) -> AnalysisTree:
        """Perform complete position analysis."""
        analysis_tree = self._build_analysis_tree(self.board.copy(), 0)

        # Close progress bar
        self.progress_bar.close()

        if analysis_tree is None:
            raise RuntimeError("Failed to analyze position: no valid moves found")

        return analysis_tree


def analyse_dfrc_position(
    params: AnalysisParams,
    engine_path: str = "stockfish",
    *,
    verbose: bool = False,
) -> AnalysisTree:
    """Perform analysis on a Chess960 position given by unique IDs."""
    chess_engine_logger = logging.getLogger("chess.engine")
    chess_engine_logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    # Get the Chess960 position
    white, black = (
        get_chess960_position(params.white_id),
        get_chess960_position(params.black_id),
    )

    # Initialize the chess board
    board = chess.Board(chess960=True)
    board.set_fen(
        f"{white.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{black.upper()} w - - 0 1",
    )
    logger.info(
        f"Analyzing position: {params.white_id=} {params.black_id=}\n{board.fen()}",
    )

    # Load the analysis configuration
    cfg = load_config(params.cfg_id)

    # Initialize the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    logger.info(f"{engine.id=}")
    if (stockfish_version := engine.id["name"]) != f"Stockfish {cfg.stockfish_version}":
        raise ValueError(
            f"Invalid Stockfish version: {stockfish_version} was used, but Stockfish {cfg.stockfish_version} is required",
        )

    # Stockfish settings
    engine.configure({"Threads": params.threads, "Hash": params.hash})
    logger.info(f"{params.threads=} {params.hash=}")

    # Initialize the RecursiveEngineAnalyzer
    analyzer = RecursiveEngineAnalyzer(board=board, engine=engine, cfg=cfg)

    # Perform analysis
    tree = analyzer.analyse()
    logger.info(
        f"Analysis complete for position: {params.white_id=} {params.black_id=}\n{board.fen()}",
    )
    # Close the engine
    engine.quit()

    return tree


if __name__ == "__main__":
    params = AnalysisParams(
        white_id=0,
        black_id=0,
        cfg_id="XS",
        threads=6,
        hash=4096,
    )
    tree = analyse_dfrc_position(
        params=params,
    )
    logger.info(f"""
          -------------------------

          Analysis tree:
            {tree}
          """)
