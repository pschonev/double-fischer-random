from collections.abc import Callable
import functools

import numpy as np
from src.utils import is_valid_chess960_position, logger


@functools.cache
def generate_pairs(
    seq: str, *, consider_duplicate_pairs: bool = True
) -> set[tuple[str, str, bool]]:
    """
    Generates a set of pairs from a given sequence.

    This function takes a string and generates a set of tuples, where each tuple represents a pair of consecutive characters in the string.
    The first tuple is always ("S", first_character) and the last tuple is always (last_character, "E"), representing the start and end of the sequence.
    If `consider_duplicate_pairs` is True, duplicate pairs are distinguished by adding a third element, 1, to the tuple.

    Parameters:
        seq: The input string from which to generate pairs.
        consider_duplicate_pairs: If True, duplicate pairs are distinguished by adding a third element, 1, to the tuple. Default is True.

    Returns:
        A set of tuples representing pairs of consecutive characters in the input string.
    """
    pairs = set()

    # this will consider edge pieces
    pairs.add(("S", seq[0]))
    pairs.add((seq[-1], "E"))

    for i in range(len(seq) - 1):
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
    """
    Calculates the local similarity between two sequences using a specified formula.

    This function generates pairs of consecutive characters from each sequence and applies the provided formula to calculate the similarity.

    Parameters:
        seq1: The first input sequence.
        seq2: The second input sequence.
        formula: A function that takes two sets of pairs and returns a float representing the similarity between the sets.

    Returns:
        The similarity between the two input sequences, as calculated by the provided formula.
    """
    pairs1 = generate_pairs(seq1)
    pairs2 = generate_pairs(seq2)

    return formula(pairs1, pairs2)


def jaccard(pairs1: set[tuple], pairs2: set[tuple]) -> float:
    """
    Calculates the Jaccard coefficient between two sets of pairs.

    The Jaccard coefficient is defined as the size of the intersection divided by the size of the union of the two sets.

    Parameters:
        pairs1: The first set of pairs.
        pairs2: The second set of pairs.

    Returns:
        The Jaccard coefficient between the two sets of pairs.
    """
    return len(pairs1.intersection(pairs2)) / len(pairs1.union(pairs2))


def sorensen_dice(pairs1: set[tuple], pairs2: set[tuple]) -> float:
    """
    Calculates the Sorensen-Dice coefficient between two sets of pairs.

    The Sorensen-Dice coefficient is defined as twice the size of the intersection divided by the sum of the sizes of the two sets.
    Note that this is idential to overleap coefficient since our sequences have the same length.

    Parameters:
        pairs1: The first set of pairs.
        pairs2: The second set of pairs.

    Returns:
        The Sorensen-Dice coefficient between the two sets of pairs.
    """
    return 2 * len(pairs1.intersection(pairs2)) / (len(pairs1) + len(pairs2))


def jaro(s1: str, s2: str) -> float:
    """
    Calculates the Jaro similarity between two strings.

    The Jaro similarity is defined as the average of the proportion of matched characters from each string and the proportion of transpositions, all divided by 3.

    Parameters:
        s1: The first input string.
        s2: The second input string.

    Returns:
        The Jaro similarity between the two input strings.
    """
    s1_matches, s2_matches = [False] * 8, [False] * 8

    matches = 0
    transpositions = 0

    # Iterate over each character in the first string
    for i in range(8):
        start = max(0, i - 3)
        end = min(i + 4, 8)

        # Look for matches in the second string within a certain range
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    # If no matches were found, return 0.0
    if matches == 0:
        return 0.0

    k = 0
    # Count transpositions
    for i in range(8):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    # Calculate Jaro similarity
    return (matches / 8 + matches / 8 + (matches - transpositions / 2) / matches) / 3


def normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Calculates the normalized Levenshtein distance between two strings.

    The Levenshtein distance is the minimum number of single-character edits (insertions, deletions or substitutions) required to change one string into the other.
    Normalization is done by dividing the Levenshtein distance by the length of the longest string.

    Parameters:
        s1: The first input string.
        s2: The second input string.

    Returns:
        The normalized Levenshtein distance between the two input strings.
    """
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
    """
    Calculates the normalized Hamming distance between two strings.

    The Hamming distance is the number of positions at which the corresponding symbols are different.
    Normalization is done by dividing the Hamming distance by the length of the strings.

    Parameters:
        s1: The first input string.
        s2: The second input string.

    Returns:
        The normalized Hamming distance between the two input strings.
    """
    seq1, seq2 = np.array(list(s1)), np.array(list(s2))
    return np.sum(seq1 != seq2) / 8


def sorensen_dice_hamming(s1: str, s2: str) -> float:
    """
    Combines Sorensen-Dice and Hamming distance to calculate a combined similarity score.

    This function calculates the Sorensen-Dice similarity and the normalized Hamming distance between two strings,
    and returns the average of these two values.

    Parameters:
        s1: The first input string.
        s2: The second input string.

    Returns:
        The combined similarity score between the two input strings.
    """
    return (
        1 - local_similarity(s1, s2, sorensen_dice) + (normalized_hamming(s1, s2))
    ) / 2


def weighted_score(score1: float, score2: float) -> float:
    """
    Calculates a weighted score based on two input scores.

    The weighted score is calculated as the sum of the squares of the two scores, divided by the sum of the two scores.
    This method of calculating a weighted score emphasizes larger values over smaller ones, which can be useful in situations where higher scores are considered significantly more valuable or impactful than lower scores.
    This is in contrast to a simple average ((score1 + score2) / 2), which treats all scores equally regardless of their magnitude.

    Parameters:
        score1: The first input score.
        score2: The second input score.

    Returns:
        The calculated weighted score.
    """
    return (score1**2 + score2**2) / (score1 + score2)


if __name__ == "__main__":
    sequence_pairs = [
        ("rqknbbnr", "nbrkbrqn", "random"),
        ("rbnqknbr", "bbrkqnrn", "random"),
        ("rnbkrqnb", "brkbnnrq", "random"),
        ("rnbknrqb", "rqbknrnb", "1-swapped"),
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
        if not is_valid_chess960_position(sequence1) or not is_valid_chess960_position(
            sequence2
        ):
            raise ValueError("Invalid sequence")  # noqa: EM101

        jaccard_score = local_similarity(sequence1, sequence2, jaccard)
        sorensen_dice_score = local_similarity(sequence1, sequence2, sorensen_dice)
        hamming_score = normalized_hamming(sequence1, sequence2)
        jaro_score = jaro(sequence1, sequence2)
        levenshtein_score = normalized_levenshtein(sequence1, sequence2)

        logger.info(
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
