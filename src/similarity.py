from functools import cache
from tabnanny import check
from typing import Callable, Tuple
from utils import logger, is_valid_chess960
import numpy as np


@cache
def generate_pairs(
    seq: str, consider_duplicate_pairs: bool = True
) -> set[Tuple[str, str, bool]]:
    """Generates set"""
    # first tuple is (S, {first_element}) and last tuple is ({last_element}, E)
    pairs = set()
    # this will consider edge pieces
    pairs.add(("S", seq[0]))
    pairs.add((seq[-1], "E"))

    for i in range(0, len(seq) - 1):
        # this distinguishes between one or two occurences of a pair
        pair = (seq[i], seq[i + 1])
        if consider_duplicate_pairs and pair in pairs:
            pairs.add((seq[i], seq[i + 1], 1))
        else:
            pairs.add(pair)
    return pairs


def local_similarity(
    seq1: str, seq2: str, formula: Callable[[set[tuple], set[tuple]], float]
) -> float:
    pairs1 = generate_pairs(seq1)
    pairs2 = generate_pairs(seq2)

    return formula(pairs1, pairs2)


def jaccard(pairs1: set[tuple], pairs2: set[tuple]) -> float:
    return len(pairs1.intersection(pairs2)) / len(pairs1.union(pairs2))


def sorensen_dice(pairs1: set[tuple], pairs2: set[tuple]) -> float:
    # this is idential to overleap coefficient since our sequences have the same length
    return 2 * len(pairs1.intersection(pairs2)) / (len(pairs1) + len(pairs2))


def jaro(s1: str, s2: str) -> float:
    s1_matches, s2_matches = [False] * 8, [False] * 8

    matches = 0
    transpositions = 0

    for i in range(8):
        start = max(0, i - 3)
        end = min(i + 4, 8)

        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(8):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / 8 + matches / 8 + (matches - transpositions / 2) / matches) / 3


def normalized_levenshtein(s1: str, s2: str) -> float:
    len_s1 = len(s1)
    len_s2 = len(s2)

    # Ensure s1 is the shorter string for space optimization
    if len_s1 > len_s2:
        s1, s2 = s2, s1
        len_s1, len_s2 = len_s2, len_s1

    # Initialize only two rows of the matrix
    previous_row = list(range(len_s2 + 1))
    current_row = [0] * (len_s2 + 1)

    # Fill in the rest of the matrix
    for i in range(1, len_s1 + 1):
        current_row[0] = i
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            current_row[j] = min(
                previous_row[j] + 1,  # Deletion
                current_row[j - 1] + 1,  # Insertion
                previous_row[j - 1] + cost,
            )  # Substitution

        # Swap current row with previous row for the next iteration
        previous_row, current_row = current_row, previous_row

    # Return the normalized Levenshtein distance
    return previous_row[len_s2] / max(len_s1, len_s2)


def normalized_hamming(s1: str, s2: str) -> float:
    seq1, seq2 = np.array(list(s1)), np.array(list(s2))
    return np.sum(seq1 != seq2) / 8


def jaro_hamming(s1: str, s2: str) -> float:
    return (jaro(s1, s2) + (1 - normalized_hamming(s1, s2))) / 2


def weighted_score(score1: float, score2: float) -> float:
    return (score1**2 + score2**2) / (score1 + score2)


if __name__ == "__main__":
    # Example usage
    sequence1 = "rrqknbkq"
    sequence2 = "nbnbrrkq"

    sequence_pairs = [
        ("rqknbbnr", "nbrkbrqn", "random"),
        ("rbnqknbr", "bbrkqnrn", "random"),
        ("rnbbnkqr", "rnbbknqr", "1-swapped"),
        ("rbnkqnbr", "rbnknqbr", "1-swapped"),  # same score if swapping on edge
        ("rbnkqnbr", "rkqbnnbr", "2-swapped"),
        ("nqbrkbrn", "brnqkbrn", "2-swapped"),
        ("rkrqbbnn", "bbnnrkrq", "4-swapped"),
        ("nbnqrkbr", "rkbrnbnq", "4-swapped"),
        ("brnqkbnr", "rnbkqnrb", "reversed"),
        ("qrnkbbnr", "rnbbknrq", "reversed"),
        ("rnbqkrnb", "brnbqkrn", "shifted by one"),
        ("rqkrbnnb", "brqkrbnn", "shifted by one"),
        ("rnbkqbnr", "rnbkqbnr", "identical"),
    ]

    for sequence1, sequence2, description in sequence_pairs:
        if not is_valid_chess960(sequence1) or not is_valid_chess960(sequence2):
            raise ValueError("Invalid sequence")

        jaccard_score = local_similarity(sequence1, sequence2, jaccard)
        sorensen_dice_score = local_similarity(sequence1, sequence2, sorensen_dice)
        hamming_score = normalized_hamming(sequence1, sequence2)
        jaro_score = jaro(sequence1, sequence2)
        levenshtein_score = normalized_levenshtein(sequence1, sequence2)

        print(
            f"""
        {sequence1} - {sequence2} ({description})
        -----------------
        Jaccard: {jaccard_score:.2f}
        Sorensen-Dice: {sorensen_dice_score:.2f}
        Hamming: {1-hamming_score:.2f}
        Jaro: {jaro_score:.2f}
        Levenshtein: {1-levenshtein_score:.2f}
        -----------------
        Jaro + Sorensen-Dice: {(jaro_score + sorensen_dice_score) / 2:.2f} (biased: {weighted_score(jaro_score, sorensen_dice_score):.2f})
        Hamming + Soresen-Dice: {(1-hamming_score + sorensen_dice_score) / 2:.2f} (biased: {weighted_score(1-hamming_score, sorensen_dice_score):.2f})
        (Jaro + Sorensen-Dice) + Hamming: {((jaro_score + sorensen_dice_score) / 2 + (1-hamming_score)) / 2:.2f} (biased: {weighted_score(weighted_score(jaro_score, sorensen_dice_score), 1-hamming_score):.2f})
        """
        )
