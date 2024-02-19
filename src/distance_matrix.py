import numpy as np
from positions import generate_positions
from similarity import normalized_hamming


class DistanceMatrix:
    def __init__(self, index):
        self.index = index
        self.matrix = np.full((len(index), len(index)), np.nan, dtype=float)

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


# the index is a dictionary that maps the position in format 'rnbqkbnr' to an integer
index = {"".join(pos): i for i, pos in enumerate(generate_positions())}
# convert strings in position to numpy arrays
position_arrays = np.array([np.array(list(pos)) for pos in generate_positions()])

distances = DistanceMatrix(index)
distances.matrix = np.array(
    [[normalized_hamming(p1, p2) for p2 in position_arrays] for p1 in position_arrays]
)
print(distances.matrix)
print((position_arrays[:, None] != position_arrays).sum(axis=2) / 8)
