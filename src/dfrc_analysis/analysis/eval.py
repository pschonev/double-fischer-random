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


def split_nodes_by_color(
    nodes: list[TreeNode],
) -> tuple[list[TreeNode], list[TreeNode]]:
    """Split nodes into white and black moves using nested set IDs."""
    white_nodes = [n for n in nodes if n.lft % 2 == 1]
    black_nodes = [n for n in nodes if n.lft % 2 == 0]
    return white_nodes, black_nodes


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
    nodes: list[TreeNode],
    min_nodes: int,
    max_nodes: int,
) -> float | None:
    """Calculate sharpness score for one color."""
    if len(nodes) == 0:
        return None  # No playable moves
    if len(nodes) <= min_nodes:
        return 1.0  # Only one line playable
    return calculate_sharpness_ratio(len(nodes), max_nodes)


def calculate_position_sharpness(
    nodes: list[TreeNode],
    cfg: AnalysisConfig,
) -> Sharpness:
    """Calculate sharpness scores for white, black and combined position."""
    balanced_nodes = filter_balanced_nodes(nodes, cfg.balanced_threshold)
    white_nodes, black_nodes = split_nodes_by_color(balanced_nodes)

    white_max, black_max = calculate_max_nodes_per_color(
        cfg.analysis_depth_ply,
        cfg.num_top_moves_per_ply,
    )

    white_min, black_min = calculate_min_nodes_per_color(cfg.analysis_depth_ply)

    white_sharpness = calculate_color_sharpness(white_nodes, white_min, white_max)
    black_sharpness = calculate_color_sharpness(black_nodes, black_min, black_max)

    total_sharpness = None
    if white_sharpness is not None and black_sharpness is not None:
        total_sharpness = harmonic_mean(white_sharpness, black_sharpness)

    return Sharpness(
        white=white_sharpness,
        black=black_sharpness,
        total=total_sharpness,
    )
