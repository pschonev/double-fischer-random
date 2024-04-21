from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from positions import generate_positions
from similarity import sorensen_dice_hamming
from utils import logger

ChessColor = Literal["white", "black"]


class DistanceMatrix:
    """A class to represent a distance matrix between all possible white/black positions.

    The class can be initialized from a DataFrame, a similarity function, or a parquet file.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @classmethod
    def from_similarity_func(
        cls, positions: list[str], similarity_func: Callable[[str, str], float]
    ) -> Self:
        """Create a distance matrix from a similarity function.

        Args:
            positions: A list of all possible positions
            similarity_func: A similarity function

        Returns:
            A DistanceMatrix instance
        """
        # create a DataFrame with the cartesian product of the positions
        distance_df = pd.MultiIndex.from_product(
            [positions, positions], names=["white", "black"]
        ).to_frame(index=False)

        distance_df["distance"] = distance_df.apply(
            lambda x: similarity_func(x["white"], x["black"]), axis=1
        )
        distance_df["analyzed"] = False
        distance_df["mirror"] = distance_df["white"] == distance_df["black"]
        distance_df["reverse"] = distance_df["white"].str[::-1] == distance_df["black"]

        distance_df = distance_df.sort_values(by="distance").reset_index(drop=True)
        return cls(distance_df)

    @classmethod
    def from_parquet(cls, file_path: Path) -> Self:
        """Create a distance matrix from a parquet file.

        Args:
            file_path: The path to the parquet file

        Returns:
            A DistanceMatrix instance
        """
        distance_df = pd.read_parquet(file_path)
        return cls(distance_df)

    def get_distances_for_positions(
        self, positions: pd.Series | list[str], color: ChessColor
    ) -> pd.DataFrame:
        """Get all distances for a list of white/black positions, exluding mirror positions.

        Args:
            positions: A list of white/black positions
            color: Black or white

        Returns:
            pd.DataFrame: A DataFrame with the distances for the positions
        """
        return self.df[(self.df[color].isin(positions)) & (self.df["mirror"] is False)][
            ["white", "black", "distance"]
        ]

    def get_sum_over_distances(
        self, positions: pd.Series | list[str], color: ChessColor
    ) -> pd.DataFrame:
        """Get the sum of distances for a list of white/black positions.

        Args:
            positions: A list of white/black positions
            color: Black or white

        Returns:
            pd.DataFrame: A DataFrame with the distances for the positions
        """
        opposite_color = "black" if color == "white" else "white"
        return (
            self.get_distances_for_positions(positions, color)
            .groupby(opposite_color)
            .sum()
            .drop(columns=[color])
            .reset_index()
            .set_index(opposite_color)
        )


if __name__ == "__main__":
    index = ["".join(pos) for pos in generate_positions()]
    distances = DistanceMatrix.from_similarity_func(
        positions=index, similarity_func=sorensen_dice_hamming
    )

    logger.info(distances.df)
    logger.info(distances.get_distances_for_positions(["rnbqkbnr"], "white"))
    sums = distances.get_sum_over_distances(
        ["rnbqkbnr", "rnbqkbrn", "rnbqknrb"], "white"
    )
    logger.info(sums)
    distances.df.to_parquet("distances.parquet")
    distances.df.to_csv("distances.csv")
