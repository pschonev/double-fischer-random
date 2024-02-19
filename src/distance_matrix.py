import json
from typing import Callable

import numpy as np
from positions import generate_positions
from similarity import jaro_hamming


class DistanceMatrix:
    def __init__(
        self, index: dict[str, int], similarity_func: Callable[[str, str], float]
    ):
        self.index = index
        self.matrix = np.array(
            [[similarity_func(p1, p2) for p2 in index.keys()] for p1 in index.keys()]
        )

    def set_value(self, row_key, col_key, value):
        try:
            self.matrix[self.index[row_key], self.index[col_key]] = value
        except KeyError:
            raise ValueError("One or both keys not found in the index.")

    def get_value(self, row_key, col_key):
        try:
            return self.matrix[self.index[row_key], self.index[col_key]]
        except KeyError:
            raise ValueError("One or both keys not found in the index.")


if __name__ == "__main__":
    # the index is a dictionary that maps the position in format 'rnbqkbnr' to an integer index
    index = {"".join(pos): i for i, pos in enumerate(generate_positions())}

    distances = DistanceMatrix(index=index, similarity_func=jaro_hamming)
    print(distances.matrix)

    # save matrix to JSON
    # together with the index
    with open("distances.json", "w") as f:
        json.dump({"index": index, "matrix": distances.matrix.tolist()}, f)
