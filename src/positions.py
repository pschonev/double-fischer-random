from itertools import combinations
from typing import Generator


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
