from typing import Self

import chess.engine
import msgspec

from src.analysis_config import AnalysisConfig, load_config
from src.positions import (
    get_chess960_position,
    chess960_to_dfrc_uid,
    is_mirrored,
    is_flipped,
)
from src.utils import harmonic_mean


class PositionAnalysis(msgspec.Struct):
    cpl: int | None
    mate: int | None
    pv: list[str] | None


class PositionNode(msgspec.Struct):
    """Node in the position analysis tree which represents a halfmove (ply) and its evaluation"""

    move: str
    children: list[Self]
    analysis: PositionAnalysis


class AnalysisData(msgspec.Struct):
    """Contains the minimum data for an analysis"""

    white_id: int
    black_id: int

    analyzer: str  # github user

    cfg_id: str

    analysis_tree: PositionNode


class Sharpness(msgspec.Struct):
    """
    Contains the sharpness of a position.
    A `None` value indicates that no good moves were available.
    """

    white: float | None
    black: float | None
    total: float | None


class AnalysisResult(AnalysisData):
    """Contains the result of an analysis with the scores.

    Attributes:
        playability_score: The harmonic mean of the sharpness_score and the win_loss_ratio
    """

    dfrc_id: int

    white_id: int
    black_id: int

    white: str
    black: str

    cfg: AnalysisConfig

    analysis_starting_pos: PositionAnalysis

    sharpness: Sharpness
    playability_score: float | None

    mirrored: bool
    flipped: bool

    @classmethod
    def from_analysis_data(cls, data: AnalysisData) -> Self:
        """Create an AnalysisResult from an AnalysisData instance"""
        cfg = load_config(data.cfg_id)

        white = get_chess960_position(data.white_id)
        black = get_chess960_position(data.black_id)

        root_analysis = data.analysis_tree.analysis

        balance_score = cls._calculate_balance_score(
            root_analysis.cpl, root_analysis.mate
        )
        sharpness = cls._calculate_sharpness_score(
            data.analysis_tree,
            cfg.balanced_threshold,
        )
        if sharpness.total is None:
            playability_score = None
        else:
            playability_score = harmonic_mean(
                balance_score,
                sharpness.total,
            )

        return cls(
            dfrc_id=chess960_to_dfrc_uid(data.white_id, data.black_id),
            white_id=data.white_id,
            black_id=data.black_id,
            white=white,
            black=black,
            cfg_id=data.cfg_id,
            cfg=cfg,
            analyzer=data.analyzer,
            analysis_tree=data.analysis_tree,
            analysis_starting_pos=PositionAnalysis(
                cpl=root_analysis.cpl,
                mate=root_analysis.mate,
                pv=root_analysis.pv,
            ),
            playability_score=playability_score,
            sharpness=sharpness,
            mirrored=is_mirrored(white, black),
            flipped=is_flipped(white, black),
        )

    @staticmethod
    def _calculate_balance_score(cpl: int | None, mate: int | None) -> float:
        """Calculate the balance score from the win, draw, loss probabilities

        Args:
            wdl: The win, draw, loss probabilities

        Returns:
            A float between 0 and 1 representing the balance of the position.
            0 is perfectly balanced, 1 is completely unbalanced.
        """
        if mate is None and cpl is None:
            raise ValueError("At least one of mate or cpl must be provided")
        if cpl is None:
            return 1
        wdl = chess.engine.Cp(cpl).wdl()
        return abs(wdl.wins - wdl.losses) / 1000

    @staticmethod
    def _calculate_sharpness_score(
        analysis_tree: PositionNode,
        balanced_threshold: int,
    ) -> Sharpness:
        """Calculate the sharpness score of a position.

        A branch (i.e. a decision point) contributes to the sharpness only if there is at least one
        balanced (good) move. If no moves are good (or no moves are analyzed), we return None so the
        absence of good moves is not conflated with a branch that is just non-forcing.

        For branches with balanced moves:
          - if there is exactly one move available (or one balanced move), the branch is considered
            maximally forcing (score 1.0),
          - if all available moves are good the branch yields 0 sharpness,
          - intermediate cases are scaled linearly.

        Args:
            analysis_tree: The root of the analysis tree.
            balanced_threshold: Moves with |cpl| less than or equal to this threshold are deemed balanced.

        Returns:
            A Sharpness object with sharpness scores for white, black, and their combined harmonic mean,
            or None if no good moves were found.
        """

        class SharpnessAccumulator(msgspec.Struct):
            balanced_moves: int = 0
            total_positions: int = 0

        white_acc = SharpnessAccumulator()
        black_acc = SharpnessAccumulator()

        def _calculate_sharpness(
            node: PositionNode,
            white: bool,
            ply: int = 0,
        ) -> None:
            if not node.children:
                return

            balanced_moves = sum(
                1
                for child in node.children
                if child.analysis.cpl is not None
                and abs(child.analysis.cpl) <= balanced_threshold
            )

            # The children of the current node represent moves available to the opponent.
            acc = black_acc if white else white_acc
            acc.balanced_moves += balanced_moves
            acc.total_positions += len(node.children)

            # Continue recursively only on moves that are balanced.
            for child in node.children:
                if (
                    child.analysis.cpl is not None
                    and abs(child.analysis.cpl) <= balanced_threshold
                ):
                    _calculate_sharpness(child, not white, ply + 1)

        _calculate_sharpness(analysis_tree, white=True)

        def calculate_final_score(acc: SharpnessAccumulator) -> float | None:
            if acc.total_positions == 0 or acc.balanced_moves == 0:
                return None  # Special case: no moves analyzed or no good moves found.
            if acc.total_positions == 1 or acc.balanced_moves == 1:
                return 1.0
            return 1.0 - ((acc.balanced_moves - 1) / (acc.total_positions - 1))

        white_sharpness = calculate_final_score(white_acc)
        black_sharpness = calculate_final_score(black_acc)

        if white_sharpness is None or black_sharpness is None:
            combined_sharpness = None
        else:
            combined_sharpness = harmonic_mean(white_sharpness, black_sharpness)

        return Sharpness(
            white=white_sharpness, black=black_sharpness, total=combined_sharpness
        )
