from dataclasses import dataclass
import chess
import chess.engine
from src.analysis_config import AnalysisConfig, load_config
from src.analysis_results import PositionNode, PositionAnalysis


@dataclass
class RecursiveEngineAnalyzer:
    board: chess.Board
    engine: chess.engine.SimpleEngine
    cfg: AnalysisConfig
    only_terminal_pv: bool = True

    def _get_candidates(
        self, board: chess.Board, ply: int
    ) -> list[chess.engine.InfoDict]:
        """Get engine analysis candidates for the given board state."""
        candidates = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.cfg.stockfish_depth_per_ply[ply]),
            multipv=self.cfg.num_top_moves_per_ply[ply],
            info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
        )
        return candidates

    def _compute_eval(
        self, score: chess.engine.PovScore
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
    ) -> PositionNode:
        """Perform complete position analysis."""
        analysis_tree = self._build_analysis_tree(self.board.copy(), 0)
        if analysis_tree is None:
            raise RuntimeError("Failed to analyze position: no valid moves found")

        return analysis_tree


if __name__ == "__main__":
    # Initialize the chess board with a sample position
    board = chess.Board()

    # Initialize the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")

    # Create an AnalysisConfig object with desired settings
    cfg = load_config("XS")

    print(engine.id)

    # Initialize the RecursiveEngineAnalyzer
    analyzer = RecursiveEngineAnalyzer(board=board, engine=engine, cfg=cfg)

    # Perform analysis (example usage)
    tree = analyzer.analyse()
    print(f"""
          
          -------------------------

          Analysis tree:
            {tree}
          """)

    # Close the engine
    engine.quit()
