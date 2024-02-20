from itertools import combinations
from typing import Generator
from utils import logger


def generate_positions() -> Generator[list[str], None, None]:
    # place bishops on opposite colors
    for bishop_a in range(0, 8, 2):
        for bishop_b in range(1, 8, 2):
            bishop_positions = {i for i in range(8)} - {bishop_a, bishop_b}
            # place knights
            for knight_a, knight_b in combinations(bishop_positions, 2):
                bishop_knight_positions = bishop_positions - {knight_a, knight_b}
                # place queen
                for queen in bishop_knight_positions:
                    starting_position = ["r"] * 8

                    # Assign the positions of the Bishops, Knights, and Queen
                    starting_position[bishop_a] = "b"
                    starting_position[bishop_b] = "b"
                    starting_position[knight_a] = "n"
                    starting_position[knight_b] = "n"
                    starting_position[queen] = "q"

                    # Find the remaining position and place the King there
                    starting_position[
                        starting_position.index("r", starting_position.index("r") + 1)
                    ] = "k"
                    yield starting_position


def is_valid_chess960_position(sequence: str) -> bool:
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
