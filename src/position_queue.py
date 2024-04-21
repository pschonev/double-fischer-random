"""Define queuing strategies."""

from pathlib import Path

import pandas as pd
from distance_matrix import ChessColor, DistanceMatrix

from src.utils import logger


def get_weighted_random_sample(distance_matrix: DistanceMatrix) -> tuple[str, str]:
    """Get a random sample of a white and black position weighted by the distance between them.

    Args:
        distance_matrix: The distance matrix

    Returns:
        The white and black positions
    """
    sample = distance_matrix.df[
        distance_matrix.df.analyzed is False & distance_matrix.df.mirror is False
    ].sample(n=1, weights=distance_matrix.df.distance, random_state=42)
    return sample.white.iloc[0], sample.black.iloc[0]


def get_max_distance_sample(distance_matrix: DistanceMatrix) -> tuple[str, str]:
    """Get the white and black position with the maximum distance between them.

    Args:
        distance_matrix: The distance matrix

    Returns:
        The white and black positions
    """
    sorted_df = (
        distance_matrix.df[
            distance_matrix.df.analyzed is False & distance_matrix.df.mirror is False
        ]
        .sort_values(by="distance", ascending=False)
        .iloc[0]
    )
    return (sorted_df["white"], sorted_df["black"])


def weighted_random_sample_from_diverse_position(
    distance_matrix: DistanceMatrix, color: ChessColor
) -> tuple[str, str]:
    """Get the white position that is most different from all analyzed white (or black) positions and then sample a black (or white) position weighted by asymmetry distance.

    Args:
        distance_matrix: The distance matrix
        color: The color to sample (white or black)

    Returns:
        The white and black positions
    """
    most_diverse_position = (
        distance_matrix.get_sum_over_distances(
            distance_matrix.df[distance_matrix.df.analyzed is True][color], color
        )
        .idxmax()
        .to_numpy()[0]
    )
    df_unanalyzed_positions = distance_matrix.df[
        (distance_matrix.df[color] == most_diverse_position)
        & (distance_matrix.df.analyzed == False)  # noqa: E712
    ]
    position = df_unanalyzed_positions.sample(
        n=1, weights=df_unanalyzed_positions.distance, random_state=42
    )
    return position.white.iloc[0], position.black.iloc[0]


def most_diverse(
    distance_matrix: DistanceMatrix, *, stochastic: bool
) -> tuple[str, str]:
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

    distance_df = distance_matrix.df.merge(
        diversity_df, on=["white", "black"], how="outer"
    )
    distance_df = distance_df[
        (distance_df.analyzed is False) & (distance_df.mirror is False)
    ]

    # Get the index of the maximum value (if stochastic, sample weighted by the value)
    if stochastic:
        sample = distance_df.sample(
            n=1, weights=distance_df.diversity_score, random_state=42
        ).iloc[0]
        return (sample["white"], sample["black"])
    else:
        sorted_df = distance_df.sort_values(by="diversity_score", ascending=False).iloc[
            0
        ]
        return (sorted_df["white"], sorted_df["black"])


if __name__ == "__main__":
    distances = DistanceMatrix.from_parquet(Path("distances.parquet"))

    distances.df.loc[distances.df.index[:5], "analyzed"] = True

    logger.info(weighted_random_sample_by_distance(distances, stochastic=True))
    logger.info(weighted_random_sample_from_diverse_position(distances, "white"))
    logger.info(most_diverse(distances, stochastic=True))
