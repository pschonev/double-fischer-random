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


def calculate_max_possible_nodes(
    depth: int,
    moves_per_ply: list[int],
    start_ply: int = 0,
) -> int:
    """Calculate theoretical maximum number of nodes for a given starting ply."""
    total = 0
    current_product = 1
    for ply in range(start_ply, depth, 2):
        current_product *= moves_per_ply[ply]
        total += current_product
    return total


def get_balanced_nodes(
    nodes: list[TreeNode],
    balanced_threshold: int,
    is_white: bool,
) -> int:
    """Count balanced nodes for a given color."""
    expected_parity = 1 if is_white else 0  # White moves are odd levels, black even
    return sum(
        1
        for node in nodes
        if node.cpl is not None
        and abs(node.cpl) <= balanced_threshold
        and (node.lft % 2 == expected_parity)
    )


def calculate_sharpness_core(
    balanced_nodes: int,
    min_nodes: int,
    max_nodes: int,
) -> float | None:
    """Calculate sharpness score given node counts."""
    if balanced_nodes < min_nodes:
        return None  # Not enough balanced nodes for a complete line

    if balanced_nodes == min_nodes:
        return 1.0  # Maximum sharpness - only one line playable

    extra_balanced = balanced_nodes - min_nodes
    extra_possible = max_nodes - min_nodes
    return 1.0 - (extra_balanced / extra_possible)


def calculate_sharpness_score(
    nodes: list[TreeNode],
    cfg: AnalysisConfig,
) -> Sharpness:
    """Calculate sharpness scores for both colors and combined."""
    # Calculate max possible nodes
    white_max = calculate_max_possible_nodes(
        cfg.analysis_depth_ply,
        cfg.num_top_moves_per_ply,
        start_ply=0,
    )
    black_max = calculate_max_possible_nodes(
        cfg.analysis_depth_ply,
        cfg.num_top_moves_per_ply,
        start_ply=1,
    )

    # Count balanced nodes
    white_balanced = get_balanced_nodes(nodes, cfg.balanced_threshold, is_white=True)
    black_balanced = get_balanced_nodes(nodes, cfg.balanced_threshold, is_white=False)

    # Minimum nodes needed for a complete line
    white_min = (cfg.analysis_depth_ply + 1) // 2  # Ceiling division
    black_min = cfg.analysis_depth_ply // 2  # Floor division

    # Calculate individual sharpness scores
    white_sharpness = calculate_sharpness_core(white_balanced, white_min, white_max)
    black_sharpness = calculate_sharpness_core(black_balanced, black_min, black_max)

    # Calculate combined score
    total_sharpness = None
    if white_sharpness is not None and black_sharpness is not None:
        total_sharpness = harmonic_mean(white_sharpness, black_sharpness)

    return Sharpness(
        white=white_sharpness,
        black=black_sharpness,
        total=total_sharpness,
    )
