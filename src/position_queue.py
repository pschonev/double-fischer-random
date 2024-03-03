"""Define queuing strategies."""

from typing import Tuple
import pandas as pd
from distance_matrix import DistanceMatrix, Color
from utils import logger

logger.setLevel("DEBUG")


def weighted_random_sample_by_distance(
    distance_matrix: DistanceMatrix, stochastic: bool
) -> Tuple[str, str]:
    """Get a random sample of a white and black position weighted by the distance between them."""
    if stochastic:
        sample = distance_matrix.df.where(distance_matrix.df.analyzed == False).sample(  # noqa: E712
            n=1, weights=distance_matrix.df.distance, random_state=42
        )
        return sample.white.iloc[0], sample.black.iloc[0]
    else:
        sorted_df = (
            distance_matrix.df[
                distance_matrix.df.analyzed == False  # noqa: E712
            ]
            .sort_values(by="distance", ascending=False)
            .iloc[0]
        )
        return (sorted_df["white"], sorted_df["black"])


def weighted_random_sample_from_diverse_position(
    distance_matrix: DistanceMatrix, color: Color
) -> Tuple[str, str]:
    """Get the white position that is most different from all analyzed white positions and then sample a black position weighted by asymmetry distance.

    First get all analyzed white positions and their asymmetry score compared to the unanalyzed white positions.
    Then take the sum of the distances and get the maximum distance.
    This is the white position that is most different compared to all already analyzed white positions.

    Next sample a black position weighted by the distance to the chosen white position.
    """
    white_position = (
        distance_matrix.get_sum_over_distances(
            distance_matrix.df[distance_matrix.df.analyzed == True][color], color
        )
        .idxmax()
        .values[0]
    )
    df_unanalyzed_white_position = distance_matrix.df[
        (distance_matrix.df.white == white_position)
        & (distance_matrix.df.analyzed == False)  # noqa: E712
    ]
    position = df_unanalyzed_white_position.sample(
        n=1, weights=df_unanalyzed_white_position.distance, random_state=42
    )
    return position.white.iloc[0], position.black.iloc[0]


def most_diverse(
    distance_matrix: DistanceMatrix, color: Color, stochastic: bool
) -> Tuple[str, str]:
    """Get the white position that is most different from all analyzed white positions and the black position that is most different from all analyzed black positions."""
    df_analyzed = distance_matrix.df[distance_matrix.df.analyzed == True]  # noqa: E712

    white_positions_weighted = distance_matrix.get_sum_over_distances(
        df_analyzed["white"], "white"
    )
    black_positions_weighted = distance_matrix.get_sum_over_distances(
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
        white_positions_weighted.reindex(index_product.index.get_level_values(0)).values
        + black_positions_weighted.reindex(
            index_product.index.get_level_values(1)
        ).values
    )

    # Reset index to make 'column_1_index' and 'column_2_index' as columns
    diversity_df = index_product.reset_index()
    diversity_df.columns = ["white", "black", "diversity_score"]

    df = distance_matrix.df.merge(diversity_df, on=["white", "black"], how="outer")
    df = df[(df.analyzed == False) & (df.mirror == False)]

    if stochastic:
        sample = df.sample(n=1, weights=df.diversity_score, random_state=42).iloc[0]
        return (sample["white"], sample["black"])
    else:
        sorted_df = df.sort_values(by="diversity_score", ascending=False).iloc[0]
        return (sorted_df["white"], sorted_df["black"])


if __name__ == "__main__":
    distances = DistanceMatrix.from_parquet("distances.parquet")

    distances.df.loc[distances.df.index[:5], "analyzed"] = True

    print(weighted_random_sample_by_distance(distances))
    print(weighted_random_sample_from_diverse_position(distances, "white"))
    print(most_diverse(distances, "white", stochastic=True))
