def harmonic_mean(a: float, b: float) -> float:
    """Calculate the harmonic mean of two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The harmonic mean of a and b, or 0 if a + b = 0
    """
    if a + b == 0:
        return 0  # Return 0 or another appropriate value when sum is zero
    return 2 * (a * b) / (a + b)


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
