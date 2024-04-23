from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from distance_matrix import DistanceMatrix
from stockfish import Stockfish
from utils import logger


@dataclass
class AnalysisConfig:
    stockfish_depth: int
    analysis_depth: int
    num_top_moves: int
    balanced_threshold: int


@dataclass
class WDL:
    win: int
    draw: int
    loss: int


@dataclass
class TopMoveEval:
    move: str
    centipawn: int
    mate: bool

    @classmethod
    def from_dict(cls, data: dict[str, str | int | None]) -> Self:
        return cls(
            move=data["Move"],  # type: ignore[reportArgumentType]
            centipawn=data["Centipawn"],  # type: ignore[reportArgumentType]
            mate=data["Mate"] is not None,
        )


@dataclass
class MoveEval:
    fen: str
    halfmove: int
    wdl: WDL
    top_moves: list[TopMoveEval]
    analysis_cfg: AnalysisConfig

    def to_row(self) -> dict[str, Any]:
        top_moves = {f"top_move_{i}": m.move for i, m in enumerate(self.top_moves)}
        top_moves_centipawn = {
            f"top_move_{i}_centipawn": m.centipawn for i, m in enumerate(self.top_moves)
        }
        return {
            "fen": self.fen,
            "move": self.halfmove // 2 + 1,
            "turn": "white" if self.halfmove % 2 == 0 else "black",
            "win": self.wdl.win,
            "draw": self.wdl.draw,
            "loss": self.wdl.loss,
            **asdict(self.analysis_cfg),
            **top_moves,
            **top_moves_centipawn,
        }


@dataclass
class FirstMoveEval(MoveEval):
    distance: float


@dataclass
class PositionResult:
    fen: str
    wdl: WDL
    centipawn: int
    sharpness_score: float
    distance: float

    @classmethod
    def from_move_eval(
        cls,
        first_move_eval: FirstMoveEval,
        deepest_moves_eval: list[MoveEval],
        balanced_threshold: int,
    ) -> Self:
        """Create a PositionResult from a FirstMoveEval and a list of MoveEval.
        The sharpeness is calculated by getting the top moves from all the deepest analyzed moves and
        checking how many of them are below the balanced threshold.
        """
        top_moves = [
            top_move for move in deepest_moves_eval for top_move in move.top_moves
        ]
        sharpness_score = sum(
            1
            for top_move in top_moves
            if (not top_move.mate) and (top_move.centipawn < balanced_threshold)
        ) / len(top_moves)
        initial_centipawn = max(
            first_move_eval.top_moves, key=lambda x: x.centipawn
        ).centipawn
        return cls(
            fen=first_move_eval.fen,
            wdl=first_move_eval.wdl,
            centipawn=initial_centipawn,
            sharpness_score=sharpness_score,
            distance=first_move_eval.distance,
        )


def convert_starting_positions_to_fen(white: str, black: str) -> str:
    """Convert the starting positions to a FEN string.

    Args:
        white: The starting position of the white pieces
        black: The starting position of the black pieces

    Returns:
        The FEN string of the starting positions
    """
    return f"{black.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{white.upper()} w KQkq - 0 1"


@dataclass
class StockfishAnalyzer:
    stockfish: Stockfish
    distance_matrix: DistanceMatrix

    cfg: AnalysisConfig
    sample_strategies: list[Callable[[], tuple[str, str]]]

    def _get_move_analysis(self, fen: str, halfmove: int) -> MoveEval:
        self.stockfish.set_fen_position(fen)

        wdl = self.stockfish.get_wdl_stats()
        if not wdl:
            msg = "No WDL stats found"
            raise ValueError(msg)

        return MoveEval(
            fen=fen,
            halfmove=halfmove,
            wdl=WDL(*wdl),
            top_moves=[
                TopMoveEval.from_dict(move)
                for move in self.stockfish.get_top_moves(self.cfg.num_top_moves)
            ],
            analysis_cfg=self.cfg,
        )

    def analyze_moves_from_position(
        self,
        halfmove: int,
    ) -> Generator[MoveEval, None, None]:
        fen = self.stockfish.get_fen_position()
        move_eval = self._get_move_analysis(fen, halfmove)
        yield move_eval

        if halfmove < (self.cfg.analysis_depth * 2 - 1):
            for top_move in move_eval.top_moves:
                self.stockfish.set_fen_position(fen)
                self.stockfish.make_moves_from_current_position([top_move.move])
                yield from self.analyze_moves_from_position(halfmove + 1)

    def set_new_position_fen(self) -> None:
        starting_positions = self.distance_matrix.get_weighted_random_sample()
        logger.info(
            f"New position: white - {starting_positions[0]} black - {starting_positions[1]}"
        )
        fen = convert_starting_positions_to_fen(*starting_positions)
        self.stockfish.set_fen_position(fen)


if __name__ == "__main__":
    cfg = AnalysisConfig(
        stockfish_depth=30,
        analysis_depth=2,
        num_top_moves=5,
        balanced_threshold=100,
    )
    analyzer = StockfishAnalyzer(
        Stockfish(
            path="/opt/homebrew/bin/stockfish",
            depth=cfg.stockfish_depth,
            parameters={"UCI_Chess960": "true", "Hash": 2048, "Threads": 7},
        ),
        DistanceMatrix.from_parquet(Path("distances.parquet")),
        cfg=cfg,
        sample_strategies=[],
    )

    # test the analyzer by analyzing one position
    analyzer.set_new_position_fen()
    for move_eval in analyzer.analyze_moves_from_position(0):
        logger.info(move_eval)
