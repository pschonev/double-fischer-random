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


def chess960_uid(white: int, black: int, N: int = 960) -> int:
    """Maps a pair of Chess960 indices to a unique integer ID.

    This function maps a pair of Chess960 indices (w, b), each in the range
    [0, N-1], to a unique integer ID in the range [0, N^2 - 1]. It satisfies
    the following requirements:

    1.  Diagonal positions (n, n) map to n, using the IDs [0, N-1].
    2.  For each row w, the off-diagonal positions (b != w) map to a
        contiguous block of IDs within the range [N, N^2 - 1].
    3.  The mapping is a bijection (collision-free and invertible).
    4.  The mirror of a UID can be easily computed by decoding, swapping,
        and re-encoding.

    For example, with N=5, the mapping looks like this:

      b=0   b=1   b=2   b=3   b=4
    w=0   0     5     6     7     8
    w=1   9     1    10    11    12
    w=2  13    14     2    15    16
    w=3  17    18    19     3    20
    w=4  21    22    23    24     4

    Args:
        w: White's index in the range [0, N-1].
        b: Black's index in the range [0, N-1].
        N: The size of the Chess960 board (default is 960).

    Returns:
        A unique integer ID in the range [0, N^2 - 1].

    Raises:
        ValueError: If w or b are not in the range [0, N-1].
    """
    if not (0 <= white < N and 0 <= black < N):
        raise ValueError(f"Indices w and b must be in the range [0, {N-1}].")

    # diagonal of symmetric positions
    if white == black:
        return white

    # off-diagonal
    row_base = N + white * (N - 1)
    offset = black if black < white else (black - 1)
    return row_base + offset


def from_chess960_uid(uid: int, N: int = 960) -> tuple[int, int]:
    """Maps a unique integer ID back to a pair of Chess960 indices.

    This function is the inverse of `chess960_uid`. It takes a unique integer
    ID in the range [0, N^2 - 1] and returns the corresponding pair of
    Chess960 indices (w, b), each in the range [0, N-1].

    Args:
        uid: A unique integer ID in the range [0, N^2 - 1].
        N: The size of the Chess960 board (default is 960).

    Returns:
        A tuple containing White's index (w) and Black's index (b), both in
        the range [0, N-1].

    Raises:
        ValueError: If uid is not in the range [0, N^2 - 1].
    """
    if not (0 <= uid < N * N):
        raise ValueError(f"UID must be in the range [0, {N*N - 1}].")

    if uid < N:
        # Diagonal
        return (uid, uid)

    # Off-diagonal
    offset = uid - N
    white = offset // (N - 1)
    remainder = offset % (N - 1)
    black = remainder if remainder < white else (remainder + 1)
    return (white, black)



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
