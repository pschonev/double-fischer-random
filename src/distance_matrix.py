from typing import Callable
import itertools

import pandas as pd
import numpy as np
from positions import generate_positions
from similarity import sorensen_dice_hamming


class DistanceMatrix:
    def __init__(self, index: list[str], similarity_func: Callable[[str, str], float]):
        similarities = [
            similarity_func(i, j) for i, j in itertools.product(index, repeat=2)
        ]
        # Reshape the results into a DataFrame
        self.df = pd.DataFrame(
            np.array(similarities).reshape(len(index), len(index)),
            index=index,
            columns=index,
        )

    def set_value(self, row_key: str, col_key: str, value: float) -> None:
        self.df.loc[row_key, col_key] = value

    def get_value(self, row_key: str, col_key: str) -> float:
        return self.df.loc[row_key, col_key]  # type: ignore

    def get_column(self, col_key: str) -> pd.Series:
        return self.df[col_key]

    def get_row(self, row_key: str) -> pd.Series:
        return self.df.loc[row_key]

    def get_sorted_values(self) -> pd.DataFrame:
        """Convert the DataFrame to a Series with the index (stack) and the values."""
        return (
            self.df.unstack()  # Convert to long form
            .reset_index()  # Reset index to get column names
            .rename(columns={"level_0": "white", "level_1": "black", 0: "distance"})
            .sort_values(by="distance")
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    index = ["".join(pos) for pos in generate_positions()]
    distances = DistanceMatrix(index=index, similarity_func=sorensen_dice_hamming)
    sorted_values = distances.get_sorted_values()

    print(distances.df)
    print(sorted_values)
    distances.df.to_parquet("distances.parquet")
    distances.df.to_csv("distances.csv")
    sorted_values.to_parquet("sorted_values.parquet")
    sorted_values.to_csv("sorted_values.csv")
