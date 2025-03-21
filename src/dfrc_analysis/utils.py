import numpy as np


def generalized_mean(a: float, b: float, p: float = -1) -> float:
    """
    Calculate generalized mean with parameter p for scoring purposes.
    Returns 0 if any calculation would cause a division by zero or domain error.
    """
    # Handle special p values first
    if p == 0:
        try:
            return np.sqrt(a * b)
        except ValueError:  # Handles negative a*b
            return 0
    elif p == float("-inf"):
        return min(a, b)
    elif p == float("inf"):
        return max(a, b)

    # For negative p, check for zeros to avoid 0**negative
    if p < 0 and (a == 0 or b == 0):
        return 0

    # For all other cases
    try:
        return ((a**p + b**p) / 2) ** (1 / p)
    except (ZeroDivisionError, ValueError, RuntimeWarning):
        return 0


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
