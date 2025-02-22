from typing import Final
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Pre-computed knight position pairs for each value of n (0-9)
# fmt: off
KNIGHT_POSITIONS: Final[tuple[tuple[int, int], ...]] = (  
    (0, 1), (0, 2), (0, 3), (0, 4),  # n = 0-3  
    (1, 2), (1, 3), (1, 4),          # n = 4-6  
    (2, 3), (2, 4),                   # n = 7-8  
    (3, 4),                           # n = 9  
)  
# fmt: on


def get_chess960_position(scharnagl: int) -> str:
    """Convert a Chess960 position number to its string representation."""
    if not 0 <= scharnagl <= 959:
        raise ValueError(f"chess960 position index not 0 <= {scharnagl} <= 959")

    position = [""] * 8
    used = set()

    # Decode position number into piece placements
    n, bw = divmod(scharnagl, 4)
    n, bb = divmod(n, 4)
    n, q = divmod(n, 6)

    # Place bishops on opposite colored squares
    bw_file = bw * 2 + 1  # White-squared bishop
    bb_file = bb * 2  # Black-squared bishop
    position[bw_file] = position[bb_file] = "b"
    used.update((bw_file, bb_file))

    # Place queen, adjusting for occupied bishop squares
    available_for_queen = [i for i in range(8) if i not in used]
    q_file = available_for_queen[q]
    position[q_file] = "q"
    used.add(q_file)

    # Get available squares for remaining pieces
    available = [i for i in range(8) if i not in used]

    # Place knights using lookup table
    if n >= len(KNIGHT_POSITIONS):
        raise ValueError(f"Invalid knight position index: {n}")
    n1, n2 = KNIGHT_POSITIONS[n]
    if n1 >= len(available) or n2 >= len(available):
        raise ValueError(f"Invalid knight position indices: {n1}, {n2}")
    position[available[n1]] = position[available[n2]] = "n"
    used.update({available[n1], available[n2]})

    # Place rooks and king in remaining squares (RKR pattern)
    remaining = [i for i in range(8) if not position[i]]
    if len(remaining) != 3:
        raise ValueError(f"Invalid remaining squares: {remaining}")
    position[remaining[0]] = position[remaining[2]] = "r"
    position[remaining[1]] = "k"

    return "".join(position)


def get_scharnagl_number(position: str) -> int:
    """Convert a Chess960 position string to its Scharnagl number (0-959)."""
    # Get bishop positions
    b_positions = sorted(i for i, p in enumerate(position) if p == "b")
    if len(b_positions) != 2:
        raise ValueError("Invalid number of bishops")

    # Calculate bishop numbers
    # Check if first bishop is on black square (even) or white square (odd)
    if b_positions[0] % 2 == 0:  # First bishop is on black square
        bb = b_positions[0] // 2
        bw = (b_positions[1] - 1) // 2
    else:  # First bishop is on white square
        bw = b_positions[0] // 2
        bb = b_positions[1] // 2

    # Get queen position
    q_pos = position.index("q")

    # Calculate queen number (0-5)
    # Count available positions before queen, excluding bishop positions
    q = sum(1 for i in range(q_pos) if i not in b_positions)

    # Get knight positions
    n_positions = sorted(i for i, p in enumerate(position) if p == "n")
    if len(n_positions) != 2:
        raise ValueError("Invalid number of knights")

    # Calculate relative knight positions
    available = [i for i in range(8) if i not in (b_positions + [q_pos])]
    n1, n2 = sorted(available.index(pos) for pos in n_positions)

    # Find knight pattern number
    try:
        n = KNIGHT_POSITIONS.index((n1, n2))
    except ValueError:
        raise ValueError(f"Invalid knight positions: {n1}, {n2}")

    # Calculate final Scharnagl number
    scharnagl = ((n * 6 + q) * 4 + bb) * 4 + bw

    return scharnagl


def chess960_to_dfrc_uid(white: int, black: int, N: int = 960) -> int:
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


def dfrc_to_chess960_uids(uid: int, N: int = 960) -> tuple[int, int]:
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


def is_mirrored(white: str, black: str) -> bool:
    """Check if a position is symmetric"""
    return white == black


def is_flipped(white: str, black: str) -> bool:
    """Check if a position is mirrored"""
    return white == black[::-1]


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
    if len(sequence) != 8:
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


if __name__ == "__main__":
    N = 960
    for i in tqdm(range(N), desc="Testing position creation"):
        is_valid_chess960_position(get_chess960_position(i))
