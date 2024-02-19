import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def is_valid_chess960(sequence: str) -> bool:
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
