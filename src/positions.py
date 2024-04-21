from collections.abc import Generator
from itertools import combinations

from utils import logger

NUMBER_OF_PIECES = 8


def generate_positions() -> Generator[list[str], None, None]:
    """Generate all valid Chess960 starting positions.

    The function generates all valid Chess960 starting positions by placing the
    bishops on opposite colors, the knights and the queen in the remaining positions,
    and the king between the rooks.

    Yields:
        A generator of all valid Chess960 starting positions
    """
    # place bishops on opposite colors
    for bishop_a in range(0, 8, 2):
        for bishop_b in range(1, 8, 2):
            positions_without_bishop = set(range(8)) - {bishop_a, bishop_b}
            # place knights
            for knight_a, knight_b in combinations(positions_without_bishop, 2):
                positions_without_bishops_or_knights = positions_without_bishop - {
                    knight_a,
                    knight_b,
                }
                # place queen
                for queen in positions_without_bishops_or_knights:
                    starting_position = ["r"] * 8

                    # Assign the positions of the Bishops, Knights, and Queen
                    starting_position[bishop_a] = "b"
                    starting_position[bishop_b] = "b"
                    starting_position[knight_a] = "n"
                    starting_position[knight_b] = "n"
                    starting_position[queen] = "q"

                    # Assign the position of the King between the Rooks
                    starting_position[
                        starting_position.index("r", starting_position.index("r") + 1)
                    ] = "k"
                    yield starting_position


def is_valid_chess960_position(sequence: str) -> bool:
    """Check if the sequence is a valid Chess960 position.

    A valid Chess960 position has the following properties:
    - 2 bishops on opposite colors
    - the king is between the rooks
    - 2 knights and 1 queen

    The function logs each wrong property and returns False if at least one of the
    properties is not respected.

    Args:
        sequence: The sequence of pieces in the starting position

    Returns:
        True if the sequence is a valid Chess960 position, False otherwise
    """
    valid = True
    if len(sequence) != NUMBER_OF_PIECES:
        logger.error(f"Invalid sequence length {len(sequence)}")
        valid = False
    if sorted(sequence) != ["b", "b", "k", "n", "n", "q", "r", "r"]:
        logger.error(f"Invalid piece counts or pieces in sequence {sequence}")
        valid = False
    # check if there is on b on odd index and one b on even index
    if sequence.index("b") % 2 == sequence.rindex("b") % 2:
        logger.error(f"Invalid sequence, bishops on same color in {sequence}")
        valid = False
    # check if k between both r values
    if sequence.index("k") < sequence.index("r") or sequence.index(
        "k"
    ) > sequence.rindex("r"):
        logger.error(f"Invalid sequence, king not between rooks in {sequence}")
        valid = False
    return valid
