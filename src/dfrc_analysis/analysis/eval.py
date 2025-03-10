import chess.engine
import msgspec

from dfrc_analysis.analysis.config import AnalysisConfig
from dfrc_analysis.db.models import TreeNode
from dfrc_analysis.utils import harmonic_mean


def calculate_balance_score(cpl: int | None, mate: int | None) -> float:
    """Calculate the balance score from the win, draw, loss probabilities

    Args:
        cpl: Centipawn loss value
        mate: Mate in N moves value

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


class Sharpness(msgspec.Struct):
    """
    Contains the sharpness of a position.
    A `None` value indicates that no good moves were available.
    """

    white: float | None
    black: float | None
    total: float | None


def calculate_sharpness_score(
    nodes: list[TreeNode],
    cfg: AnalysisConfig,
) -> Sharpness:
    # Calculate theoretical maximum nodes for white and black
    def calculate_max_possible_nodes(*, is_white: bool) -> int:
        total = 0
        current_product = 1

        # Determine which plies belong to this player
        start_ply = 0 if is_white else 1

        for ply in range(start_ply, cfg.analysis_depth_ply, 2):
            current_product *= cfg.num_top_moves_per_ply[ply]
            total += current_product

        return total

    white_max = calculate_max_possible_nodes(is_white=True)
    black_max = calculate_max_possible_nodes(is_white=False)

    # Count balanced nodes for white and black
    white_balanced = sum(
        1
        for node in nodes
        if node.cpl is not None
        and abs(node.cpl) <= cfg.balanced_threshold
        and (node.lft % 2 == 1)  # White's moves are at odd levels in the tree
    )

    black_balanced = sum(
        1
        for node in nodes
        if node.cpl is not None
        and abs(node.cpl) <= cfg.balanced_threshold
        and (node.lft % 2 == 0)  # Black's moves are at even levels in the tree
    )

    # Calculate minimum nodes needed for a single complete line
    white_min = (cfg.analysis_depth_ply + 1) // 2  # Ceiling division for white
    black_min = cfg.analysis_depth_ply // 2  # Floor division for black

    # Calculate sharpness scores
    def calculate_score(
        balanced: int,
        min_needed: int,
        max_possible: int,
    ) -> float | None:
        if balanced < min_needed:
            return None  # Not enough balanced nodes for even one complete line

        if balanced == min_needed:
            return 1.0  # Maximum sharpness - only one line is playable

        # Calculate how many nodes beyond the minimum we have
        extra_balanced = balanced - min_needed
        extra_possible = max_possible - min_needed

        # Return inverted ratio (1.0 = max sharpness, 0.0 = all moves playable)
        return 1.0 - (extra_balanced / extra_possible)

    white_sharpness = calculate_score(white_balanced, white_min, white_max)
    black_sharpness = calculate_score(black_balanced, black_min, black_max)

    # Calculate combined sharpness using harmonic mean
    if white_sharpness is None or black_sharpness is None:
        combined_sharpness = None
    else:
        combined_sharpness = harmonic_mean(white_sharpness, black_sharpness)

    return Sharpness(
        white=white_sharpness,
        black=black_sharpness,
        total=combined_sharpness,
    )
