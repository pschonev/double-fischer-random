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
