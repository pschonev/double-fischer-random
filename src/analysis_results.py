from typing import Self

import chess.engine
import msgspec

from src.analysis_config import AnalysisConfig, load_config
from src.positions import get_chess960_position, chess960_uid, is_mirrored, is_flipped
from src.utils import harmonic_mean


class PositionNode(msgspec.Struct):
    """Node in the position analysis tree which represents a halfmove (ply) and its evaluation"""

    move: str
    cpl: int
    children: list[Self]
    pv: list[str]


class AnalysisData(msgspec.Struct):
    """Contains the minimum data for an analysis"""

    white_id: int
    black_id: int

    analyzer: str
    validator: str

    cfg_id: str

    analysis_tree: PositionNode


class Sharpness(msgspec.Struct):
    """Contains the sharpness of a position
    The total is a combination of the white and black sharpness"""

    white: float
    black: float
    total: float


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

    cpl: int
    pv: list[str]

    sharpness: Sharpness
    playability_score: float

    mirrored: bool
    flipped: bool

    @classmethod
    def from_analysis_data(cls, data: AnalysisData) -> Self:
        """Create an AnalysisResult from an AnalysisData instance"""
        cfg = load_config(data.cfg_id)

        white = get_chess960_position(data.white_id)
        black = get_chess960_position(data.black_id)

        balance_score = cls._calculate_balance_score(data.analysis_tree.cpl)
        sharpness = cls._calculate_sharpness_score(
            data.analysis_tree,
            cfg.balanced_threshold,
        )

        return cls(
            dfrc_id=chess960_uid(data.white_id, data.black_id),
            white_id=data.white_id,
            black_id=data.black_id,
            white=white,
            black=black,
            cfg_id=data.cfg_id,
            cfg=cfg,
            analyzer=data.analyzer,
            validator=data.validator,
            analysis_tree=data.analysis_tree,
            cpl=data.analysis_tree.cpl,
            pv=data.analysis_tree.pv,
            playability_score=harmonic_mean(
                balance_score,
                sharpness.total,
            ),
            sharpness=sharpness,
            mirrored=is_mirrored(white, black),
            flipped=is_flipped(white, black),
        )

    @staticmethod
    def _calculate_balance_score(cpl: int) -> float:
        """Calculate the balance score from the win, draw, loss probabilities

        Args:
            wdl: The win, draw, loss probabilities

        Returns:
            A float between 0 and 1 representing the balance of the position.
            0 is perfectly balanced, 1 is completely unbalanced.
        """
        wdl = chess.engine.Cp(cpl).wdl()
        return abs(wdl.wins - wdl.losses) / 1000

    @staticmethod
    def _calculate_sharpness_score(
        analysis_tree: PositionNode,
        balanced_threshold: int,
    ) -> Sharpness:
        """Calculate the sharpness score of a position

        A branch (i.e. a decision point) will only contribute to sharpness if there is at least one
        balanced (good) move. If no moves are good, we treat that branch as non-forcing (sharpness 0).
        Furthermore, if exactly one move is balanced, then the branch is maximally forcing (score 1),
        regardless of the total number of moves.

        For branches with more than one balanced move, the score is scaled linearly so that:
        - all moves good (x = t) gives 0 sharpness,
        - one move good (x = 1) gives 1 sharpness, and
        - intermediate cases interpolate linearly.

        Args:
            analysis_tree: The root of the analysis tree.
            balanced_threshold: Moves with |cpl| below or equal to this threshold are deemed balanced.

        Returns:
            A Sharpness object with sharpness scores for white, black, and their combination.
        """

        # Local accumulator for moves.
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
            # Stop if there are no candidate moves forwarded
            if not node.children:
                return

            # Count moves that are considered balanced/good at this decision point.
            balanced_moves = sum(
                1 for child in node.children if abs(child.cpl) <= balanced_threshold
            )

            # Note: The way your tree is built, the children of the current node represent the moves
            # available to the player who is about to move. Thus, we update the accumulator for the
            # opponent. (For example, at the root call with white=True, the children are black’s moves.)
            acc = black_acc if white else white_acc
            acc.balanced_moves += balanced_moves
            acc.total_positions += len(node.children)

            # Continue recursively only on moves that are balanced.
            for child in node.children:
                if abs(child.cpl) <= balanced_threshold:
                    _calculate_sharpness(child, not white, ply + 1)

        # Initiate recursion.
        _calculate_sharpness(analysis_tree, white=True)

        # Now, compute a final sharpness score for a given accumulator.
        def calculate_final_score(acc: SharpnessAccumulator) -> float:
            if acc.total_positions == 0:
                return 0.0  # no moves analyzed
            if acc.balanced_moves == 0:
                return 0.0  # branch with no good moves is disregarded
            # If there is exactly one balanced move at the decision point OR only one move in total,
            # that branch is maximally forcing.
            if acc.total_positions == 1 or acc.balanced_moves == 1:
                return 1.0
            # Otherwise, scale the sharpness so that:
            #   1 balanced move yields 1, and all moves balanced yields 0.
            return 1.0 - ((acc.balanced_moves - 1) / (acc.total_positions - 1))

        white_sharpness = calculate_final_score(white_acc)
        black_sharpness = calculate_final_score(black_acc)

        # Combine using the harmonic mean as before.
        if white_sharpness + black_sharpness == 0:
            combined_sharpness = 0.0
        else:
            combined_sharpness = harmonic_mean(white_sharpness, black_sharpness)

        return Sharpness(white_sharpness, black_sharpness, combined_sharpness)
