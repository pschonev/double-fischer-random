from collections import deque
import random
from collections.abc import Generator
from typing import Callable, Final

from pqdict import pqdict
from tqdm import tqdm

from src.similarity import sorensen_dice_hamming
from src.utils import is_valid_chess960_position


def calculate_symmetry_score(white: int, black: int) -> float:
    """Calculate the symmetry score between two Chess960 positions.
    Args:
        white: The index of the white position in the range [0, 959].
        black: The index of the black position in the range [0, 959].

    Returns:
        A float between 0 and 1 representing the similarity between the positions.
        1 means the positions are identical, 0 means they are completely different.
    """
    pos_white = get_chess960_position(white)
    pos_black = get_chess960_position(black)
    return 1 - sorensen_dice_hamming(pos_white, pos_black)


class RecentIndices:
    """A deque-like data structure that keeps track of the most recent indices."""

    def __init__(self, maxlen):
        self._deque = deque(maxlen=maxlen)
        self._set = set()
        self.maxlen = maxlen

    def appendleft(self, item):
        if len(self._deque) >= self.maxlen:
            removed_item = self._deque.pop()
            self._set.discard(removed_item)
        self._deque.appendleft(item)
        self._set.add(item)

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return len(self._deque)


def sample_positions(
    N: int = 960, priority_func: Callable[[int, int], float] = lambda w, b: 0.5
) -> Generator[tuple[int, int], None, None]:
    """
    Lazily generates Double Fischer Random Chess positions in order of least analysis.

    This function uses a priority queue to ensure that positions are sampled
    in the order of their analysis priority. Each position is represented as
    a tuple `(w, b)`, where `w` is the white position index and `b` is the
    black position index. The priority of a position is determined by the
    sum of the analysis counts for the corresponding white and black indices,
    with initial priorities set using a custom function.

    Args:
        N (int): The number of possible positions for each color (default: 960).
        priority_func (callable): A function that takes two integers `(w, b)`
            and returns a float between 0 and 1. This value is used as a
            probability to initialize the priority of the position.

    Yields:
        tuple[int, int]: The next position `(w, b)` to analyze, in order of least analysis.

    Example:
        >>> def custom_priority(w, b):
        ...     return (w + b) % 2 / 2  # Example priority function
        >>> generator = sample_positions(960, priority_func=custom_priority)
        >>> next(generator)
        (0, 0)
        >>> for position in generator:
        ...     print(position)
    """
    pq = pqdict()
    random.seed(42)

    recent_white_indices = RecentIndices(maxlen=N // 2)
    recent_black_indices = RecentIndices(maxlen=N // 2)

    for w in tqdm(range(N), desc="Initializing positions"):
        for b in range(N):
            # calculate an initial priority that is
            # between 0 (high priority) and 4 (low priority)
            priority_score = priority_func(w, b)  # between 0 and 1
            probabilistic_priority = priority_score + random.random() / 2
            initial_priority = probabilistic_priority * 10 // 2
            pq[(w, b)] = initial_priority

    update_threshold = N * 32
    with tqdm(total=N * N, desc="Processing positions") as pbar:
        while pq:
            (w, b), priority = pq.popitem()

            if (
                (w_recent := w in recent_white_indices)
                or (b_recent := b in recent_black_indices)
            ) and len(pq) >= update_threshold:
                pq[(w, b)] = priority + int(w_recent) + int(b_recent)
                continue

            yield (w, b)

            recent_white_indices.appendleft(w)
            recent_black_indices.appendleft(b)

            pbar.update(1)


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
    """Convert a Chess960 position string to its Scharnagl number (0-959).

    Args:
        position: An 8-character string representing piece placement (e.g., 'rnbqkbnr')

    Returns:
        Integer between 0 and 959 representing the Chess960 position

    Raises:
        ValueError: If the position is not a valid Chess960 position
    """
    # Validate the position first
    if not is_valid_chess960_position(position):
        raise ValueError(f"Invalid Chess960 position: {position}")

    # Get bishop positions
    b1, b2 = sorted(i for i, p in enumerate(position) if p == "b")
    # Convert to bishop pairs (0-3)
    bb = b1 // 2  # Black square bishop position number
    bw = (b2 - 1) // 2  # White square bishop position number

    # Get queen position and convert to number (0-5)
    q_pos = position.index("q")
    # Adjust queen number based on bishops before it
    q = q_pos
    q -= sum(1 for b in (b1, b2) if b < q_pos)

    # Get knight positions
    knight_positions = [i for i, p in enumerate(position) if p == "n"]
    # Convert to relative positions among empty squares
    available_spots = [i for i in range(8) if i not in (b1, b2, q_pos)]
    n1, n2 = sorted(available_spots.index(k) for k in knight_positions)

    # Find n value from knight positions using KNIGHT_POSITIONS lookup
    n = next(i for i, (x, y) in enumerate(KNIGHT_POSITIONS) if (x, y) == (n1, n2))

    # Calculate final Scharnagl number
    scharnagl = ((n * 6 + q) * 4 + bb) * 4 + bw

    return scharnagl


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


if __name__ == "__main__":
    N = 960
    for i in tqdm(range(N), desc="Testing position creation"):
        is_valid_chess960_position(get_chess960_position(i))

    chess960_combinations = set()
    with open("chess960_positions.txt", "w") as f:
        f.write("id,white,black\n")
        for k, (w, b) in enumerate(
            sample_positions(
                N=N,
                priority_func=calculate_symmetry_score,
            )
        ):
            dfrc_position = chess960_uid(w, b, N)
            f.write(f"{k},{w},{b}\n")
            chess960_combinations.add((w, b))
        print(
            f"Expected positions: {N*N}\nGenerated positions: {k+1}\nGenerated unique positions: {len(chess960_combinations)}"
        )
