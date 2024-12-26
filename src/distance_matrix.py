from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from src.positions import generate_positions
from src.similarity import sorensen_dice_hamming
from src.utils import logger

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

    def _get_distances_for_positions(
        self, positions: pd.Series | list[str], color: ChessColor
    ) -> pd.DataFrame:
        """Get all distances for a list of white/black positions, exluding mirror positions.

        Args:
            positions: A list of white/black positions
            color: Black or white

        Returns:
            pd.DataFrame: A DataFrame with the distances for the positions
        """
        return self.df[(self.df[color].isin(positions)) & (self.df.mirror == False)][
            ["white", "black", "distance"]
        ]

    def _get_sum_over_distances(
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
            self._get_distances_for_positions(positions, color)
            .groupby(opposite_color)
            .sum()
            .drop(columns=[color])
            .reset_index()
            .set_index(opposite_color)
        )

    def get_weighted_random_sample(self) -> tuple[str, str]:
        """Get a random sample of a white and black position weighted by the distance between them.

        Args:
            distance_matrix: The distance matrix

        Returns:
            The white and black positions
        """
        sample = self.df[
            (self.df.analyzed == False) & (self.df.mirror == False)
        ].sample(n=1, weights=self.df.distance, random_state=42)
        return sample.white.iloc[0], sample.black.iloc[0]

    def get_max_distance_sample(self) -> tuple[str, str]:
        """Get the white and black position with the maximum distance between them.

        Args:
            distance_matrix: The distance matrix

        Returns:
            The white and black positions
        """
        sorted_df = (
            self.df[(self.df.analyzed == False) & (self.df.mirror == False)]
            .sort_values(by="distance", ascending=False)
            .iloc[0]
        )
        return (sorted_df["white"], sorted_df["black"])

    def weighted_random_sample_from_diverse_position(
        self, color: ChessColor
    ) -> tuple[str, str]:
        """Get the white position that is most different from all analyzed white (or black) positions and then sample a black (or white) position weighted by asymmetry distance.

        Args:
            distance_matrix: The distance matrix
            color: The color to sample (white or black)

        Returns:
            The white and black positions
        """
        most_diverse_position = (
            self._get_sum_over_distances(
                self.df[self.df.analyzed == True][color], color
            )
            .idxmax()
            .to_numpy()[0]
        )
        df_unanalyzed_positions = self.df[
            (self.df[color] == most_diverse_position) & (self.df.analyzed == False)
        ]
        position = df_unanalyzed_positions.sample(
            n=1, weights=df_unanalyzed_positions.distance, random_state=42
        )
        return position.white.iloc[0], position.black.iloc[0]

    def get_most_diverse_position(self, *, stochastic: bool) -> tuple[str, str]:
        """Get the white position that is most different from all analyzed white positions and the black position that is most different from all analyzed black positions.

        Get all analyzed white and black positions and their asymmetry score compared to the unanalyzed positions.
        Then take the sum of the distances and get the maximum distance. This is the white position that is most different compared to all already analyzed white positions.

        If stochastic is True, sample a black position weighted by the distance to the chosen white position.

        Args:
            distance_matrix: The distance matrix
            stochastic: Whether to use a stochastic approach

        Returns:
            The white and black positions
        """
        df_analyzed = self.df[self.df.analyzed == True]

        white_positions_weighted = self._get_sum_over_distances(
            df_analyzed["white"], "white"
        )
        black_positions_weighted = self._get_sum_over_distances(
            df_analyzed["black"], "black"
        )

        # Create a DataFrame with the Cartesian product of indices
        index_product = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [white_positions_weighted.index, black_positions_weighted.index]
            )
        )
        # Perform element-wise addition for each pair of indices
        index_product["value_sum"] = (
            white_positions_weighted.reindex(
                index_product.index.get_level_values(0)
            ).to_numpy()
            + black_positions_weighted.reindex(
                index_product.index.get_level_values(1)
            ).to_numpy()
        )

        # reshape the DataFrame to have the white and black positions as columns and filter
        diversity_df = index_product.reset_index()
        diversity_df.columns = ["white", "black", "diversity_score"]

        distance_df = self.df.merge(diversity_df, on=["white", "black"], how="outer")
        distance_df = distance_df[
            (distance_df.analyzed == False) & (distance_df.mirror == False)
        ]

        # Get the index of the maximum value (if stochastic, sample weighted by the value)
        if stochastic:
            sample = distance_df.sample(
                n=1, weights=distance_df.diversity_score, random_state=42
            ).iloc[0]
            return (sample["white"], sample["black"])
        else:
            sorted_df = distance_df.sort_values(
                by="diversity_score", ascending=False
            ).iloc[0]
            return (sorted_df["white"], sorted_df["black"])


if __name__ == "__main__":
    index = ["".join(pos) for pos in generate_positions()]
    distances = DistanceMatrix.from_similarity_func(
        positions=index, similarity_func=sorensen_dice_hamming
    )
    logger.info(distances.df)
    distances.df.to_parquet("distances.parquet")
    distances.df.to_csv("distances.csv")

    # test distance matrix
    logger.info(distances._get_distances_for_positions(["rnbqkbnr"], "white"))  # noqa: SLF001
    sums = distances._get_sum_over_distances(  # noqa: SLF001
        ["rnbqkbnr", "rnbqkbrn", "rnbqknrb"], "white"
    )
    logger.info(sums)

    # test position sampling
    distances.df.loc[distances.df.index[:5], "analyzed"] = True
    logger.info(distances.get_weighted_random_sample())
    logger.info(distances.get_max_distance_sample())
    logger.info(distances.weighted_random_sample_from_diverse_position("white"))
    logger.info(distances.weighted_random_sample_from_diverse_position("black"))
    logger.info(distances.get_most_diverse_position(stochastic=True))
    logger.info(distances.get_most_diverse_position(stochastic=False))
