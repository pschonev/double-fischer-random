import random
from collections import deque
from collections.abc import Generator
from typing import Callable

from pqdict import pqdict
from tqdm import tqdm

from src.positions import get_chess960_position
from src.similarity import sorensen_dice_hamming


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


if __name__ == "__main__":
    N = 960
    chess960_combinations = set()
    with open("chess960_positions.txt", "w") as f:
        f.write("id,white,black\n")
        for k, (w, b) in tqdm(
            enumerate(
                sample_positions(
                    N=N,
                    priority_func=calculate_symmetry_score,
                ),
            ),
            desc="Sampling positions",
            total=N * N,
        ):
            f.write(f"{k},{w},{b}\n")
            chess960_combinations.add((w, b))
        print(
            f"Expected positions: {N*N}\nGenerated positions: {k+1}\nGenerated unique positions: {len(chess960_combinations)}"
        )
