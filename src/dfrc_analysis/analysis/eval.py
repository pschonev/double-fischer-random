from functools import lru_cache

import msgspec
import numpy as np

from dfrc_analysis.analysis.config import AnalysisConfig
from dfrc_analysis.db.models import TreeNode
from dfrc_analysis.utils import harmonic_mean


@lru_cache(maxsize=128)
def _calculate_normalization_factors(
    threshold: int, steepness: float
) -> tuple[float, float, float]:
    """Calculate and cache the normalization factors for the balance score function.

    Args:
        threshold: The maximum centipawn value to consider
        steepness: Controls how steep the curve is in the middle

    Returns:
        Tuple of (min_val, max_val, k) for normalization
    """
    k = steepness / threshold
    min_val = 1 / (1 + np.exp(-k * (0 - threshold / 2)))
    max_val = 1 / (1 + np.exp(-k * (threshold - threshold / 2)))
    return min_val, max_val, k


def calculate_balance_score(
    cpl: int | None,
    mate: int | None,
    threshold: int = 50,
    steepness: float = 10.0,
) -> float:
    """Calculate balance score using a normalized logistic function.

    Args:
        cpl: Centipawn loss value (None if mate is provided)
        mate: Mate in N moves value (None if cpl is provided)
        threshold: The maximum centipawn value to consider (default: 50)
        steepness: Controls how steep the curve is in the middle (default: 8.0)
                   Higher values make the middle steeper and ends flatter

    Returns:
        A float between 0 and 1 representing imbalance (0=balanced, 1=imbalanced)
    """
    # If there's a mate, return maximum imbalance
    if mate is not None:
        return 1.0

    # Ensure we have a cpl value
    if cpl is None:
        raise ValueError("At least one of mate or cpl must be provided")

    # Get cached normalization factors
    min_val, max_val, k = _calculate_normalization_factors(threshold, steepness)

    # Use absolute value of centipawn loss
    x = abs(cpl)

    # Raw logistic function
    raw = 1 / (1 + np.exp(-k * (x - threshold / 2)))

    # Normalize to exactly [0,1]
    return 1 - np.clip((raw - min_val) / (max_val - min_val), 0.0, 1.0)


class Sharpness(msgspec.Struct):
    """
    Contains the sharpness of a position.
    A `None` value indicates that no good moves were available.
    """

    white: float | None
    black: float | None
    total: float | None


def calculate_sharpness_ratio(
    balanced_nodes: int,
    max_total_nodes: int,
    power: float = 1.0,
) -> float:
    """Calculate sharpness ratio based on balanced nodes vs maximum possible nodes.

    A linear calculation (power=1.0) is used by default since non-linearity is largely
    unnecessary due to the tree structure:
    - Balanced nodes higher in the tree unlock multiple nodes below them
    - This creates an implicit weighing where higher nodes naturally have more impact
    - The multiplicative nature of the tree means adding a balanced node at depth N
      can unlock N^d nodes at deeper levels

    Args:
        balanced_nodes: Number of nodes within acceptable evaluation threshold
        max_total_nodes: Maximum theoretical number of nodes possible
        power: Optional power to adjust sensitivity (default=1.0 for linear calculation)

    Returns:
        Float between 0.0 and 1.0, where:
        - Values closer to 1.0 indicate fewer balanced nodes (sharper position)
        - Values closer to 0.0 indicate more balanced nodes (more open position)
    """
    ratio = balanced_nodes / max_total_nodes
    return 1.0 - (ratio**power)


# eval.py


def filter_balanced_nodes(nodes: list[TreeNode], threshold: int) -> list[TreeNode]:
    """Filter nodes within the balanced threshold."""
    return [n for n in nodes if n.cpl is not None and abs(n.cpl) <= threshold]


def count_nodes_by_color(
    nodes: list[TreeNode],
) -> tuple[int, int]:
    """Count nodes for white and black moves using nested set IDs."""
    white_count = sum(1 for n in nodes if n.lft % 2 == 0)
    black_count = sum(
        1 for n in nodes if n.lft % 2 == 1 and n.move != "root"
    )  # the root node is irrelevant
    return white_count, black_count


def calculate_max_nodes_per_color(
    moves_per_ply: list[int],
) -> tuple[int, int]:
    """Calculate maximum possible nodes for white and black."""
    white_total = 0
    black_total = 0
    cumulative_product = 1

    for i in range(len(moves_per_ply)):
        cumulative_product *= moves_per_ply[i]
        # White's level
        if i % 2 == 0:
            white_total += cumulative_product
        # Black's level
        else:
            black_total += cumulative_product

    return white_total, black_total


def calculate_min_nodes_per_color(depth: int) -> tuple[int, int]:
    """Calculate minimum nodes needed for a complete line per color."""
    white_min = (depth + 1) // 2  # Ceiling division
    black_min = depth // 2  # Floor division
    return white_min, black_min


def calculate_color_sharpness(
    node_count: int,
    min_nodes: int,
    max_nodes: int,
) -> float | None:
    """Calculate sharpness score for one color."""
    if node_count == 0:
        return None  # No playable moves
    if node_count <= min_nodes:
        return 1.0  # Only one line playable
    return calculate_sharpness_ratio(node_count, max_nodes)


def calculate_position_sharpness(
    nodes: list[TreeNode],
    cfg: AnalysisConfig,
) -> Sharpness:
    """Calculate sharpness scores for white, black and combined position."""
    balanced_nodes = filter_balanced_nodes(nodes, cfg.balanced_threshold)
    white_node_count, black_node_count = count_nodes_by_color(balanced_nodes)

    white_max, black_max = calculate_max_nodes_per_color(
        cfg.num_top_moves_per_ply,
    )

    white_min, black_min = calculate_min_nodes_per_color(cfg.analysis_depth_ply)

    white_sharpness = calculate_color_sharpness(white_node_count, white_min, white_max)
    black_sharpness = calculate_color_sharpness(black_node_count, black_min, black_max)

    total_sharpness = None
    if white_sharpness is not None and black_sharpness is not None:
        total_sharpness = harmonic_mean(white_sharpness, black_sharpness)

    return Sharpness(
        white=white_sharpness,
        black=black_sharpness,
        total=total_sharpness,
    )
