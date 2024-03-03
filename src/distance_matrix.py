from typing import Callable, Literal

import pandas as pd
from positions import generate_positions
from similarity import sorensen_dice_hamming

Color = Literal["white", "black"]


class DistanceMatrix:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @classmethod
    def from_similarity(
        cls, positions: list[str], similarity_func: Callable[[str, str], float]
    ):
        # create a DataFrame with the cartesian product of the positions
        df = pd.MultiIndex.from_product(
            [positions, positions], names=["white", "black"]
        ).to_frame(index=False)

        df["distance"] = df.apply(
            lambda x: similarity_func(x["white"], x["black"]), axis=1
        )
        df["analyzed"] = False
        df["mirror"] = df["white"] == df["black"]
        df["reverse"] = df["white"].str[::-1] == df["black"]

        df = df.sort_values(by="distance").reset_index(drop=True)
        return cls(df)

    @classmethod
    def from_parquet(cls, file_path: str):
        df = pd.read_parquet(file_path)
        return cls(df)

    def get_distances_for_positions(
        self, positions: pd.Series | list[str], color: Color
    ) -> pd.DataFrame:
        """Get all distances for a list of white/black positions, exluding mirror positions."""
        return self.df[(self.df[color].isin(positions)) & (self.df["mirror"] == False)][
            ["white", "black", "distance"]
        ]

    def get_sum_over_distances(
        self, positions: pd.Series | list[str], color: Color
    ) -> pd.DataFrame:
        """Get the sum of distances for a list of white/black positions."""
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
    distances = DistanceMatrix.from_similarity(
        positions=index, similarity_func=sorensen_dice_hamming
    )

    print(distances.df)
    print(distances.get_distances_for_positions(["rnbqkbnr"], "white"))
    sums = distances.get_sum_over_distances(
        ["rnbqkbnr", "rnbqkbrn", "rnbqknrb"], "white"
    )
    print(sums)
    distances.df.to_parquet("distances.parquet")
    distances.df.to_csv("distances.csv")
