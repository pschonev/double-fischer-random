import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def harmonic_mean(a: float, b: float) -> float:
    """Calculate the harmonic mean of two numbers"""
    return 2 * (a * b) / (a + b)
