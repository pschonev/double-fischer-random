import logging
from dataclasses import dataclass

import chess
import chess.engine
from tqdm import tqdm

from src.analysis_config import AnalysisConfig, load_config
from src.analysis_results import AnalysisParams, PositionAnalysis, PositionNode
from src.positions import get_chess960_position

AnalysisTree = PositionNode

logger = logging.getLogger(__name__)


@dataclass
class RecursiveEngineAnalyzer:
    board: chess.Board
    engine: chess.engine.SimpleEngine
    cfg: AnalysisConfig
    only_terminal_pv: bool = True

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

    def _build_analysis_tree(self, board: chess.Board, ply: int) -> PositionNode | None:
        """Recursively build the analysis tree."""
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
