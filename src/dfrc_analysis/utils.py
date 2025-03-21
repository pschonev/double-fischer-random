import numpy as np


def generalized_mean(a: float, b: float, p: float = -1) -> float:
    """
    Calculate the generalized mean with parameter p.

    Special cases:
    - p = 1: Arithmetic mean
    - p = 0: Geometric mean (limit as p approaches 0)
    - p = -1: Harmonic mean
    - p = -∞: Minimum value
    - p = +∞: Maximum value

    Lower p values increase sensitivity to small values.
    """
    if p == 0:
        return np.sqrt(a * b)
    elif p == float("-inf"):
        return min(a, b)
    elif p == float("inf"):
        return max(a, b)
    else:
        return ((a**p + b**p) / 2) ** (1 / p)


def calculate_subtree_size(
    start_ply: int,
    analysis_depth_ply: int,
    num_top_moves_per_ply: list[int],
) -> int:
    """
    Calculate the size of a subtree starting from a given ply.
    When start_ply=0, this calculates the total size of the full analysis tree.
    """
    if start_ply >= analysis_depth_ply:
        return 0

    total = 1
    product = 1

    for ply in range(start_ply, analysis_depth_ply):
        product *= num_top_moves_per_ply[ply]
        total += product

    return total
