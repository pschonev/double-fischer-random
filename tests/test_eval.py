import pytest

from dfrc_analysis.analysis.config import AnalysisConfig
from dfrc_analysis.analysis.eval import (
    Sharpness,
    TreeNode,
    calculate_color_sharpness,
    calculate_max_nodes_per_color,
    calculate_min_nodes_per_color,
    calculate_position_sharpness,
    calculate_sharpness_ratio,
    filter_balanced_nodes,
    split_nodes_by_color,
)
from dfrc_analysis.db.models import TreeNode
from dfrc_analysis.utils import harmonic_mean


def test_calculate_sharpness_ratio():
    assert calculate_sharpness_ratio(0, 10) == 1.0
    assert calculate_sharpness_ratio(10, 10) == 0.0
    assert calculate_sharpness_ratio(5, 10) == 0.5
    assert calculate_sharpness_ratio(5, 10, power=2) == 0.75


def test_filter_balanced_nodes():
    nodes = [
        TreeNode(cpl=10, lft=1, rgt=2, dfrc_id=1, cfg_id="1", move="e2e4"),
        TreeNode(cpl=30, lft=3, rgt=4, dfrc_id=1, cfg_id="1", move="d2d4"),
        TreeNode(cpl=15, lft=5, rgt=6, dfrc_id=1, cfg_id="1", move="g1f3"),
        TreeNode(cpl=None, lft=7, rgt=8, dfrc_id=1, cfg_id="1", move="c2c4"),
    ]
    result = filter_balanced_nodes(nodes, 20)
    assert len(result) == 2
    assert result[0].cpl == 10
    assert result[1].cpl == 15


def test_split_nodes_by_color():
    nodes = [
        TreeNode(cpl=10, lft=1, rgt=2, dfrc_id=1, cfg_id="1", move="e2e4"),
        TreeNode(cpl=20, lft=2, rgt=3, dfrc_id=1, cfg_id="1", move="e7e5"),
        TreeNode(cpl=15, lft=3, rgt=4, dfrc_id=1, cfg_id="1", move="g1f3"),
        TreeNode(cpl=25, lft=4, rgt=5, dfrc_id=1, cfg_id="1", move="b8c6"),
    ]
    white_nodes, black_nodes = split_nodes_by_color(nodes)
    assert len(white_nodes) == 2
    assert len(black_nodes) == 2
    assert white_nodes[0].move == "e2e4"
    assert black_nodes[0].move == "e7e5"


def test_calculate_max_nodes_per_color():
    moves_per_ply = [2, 3, 2]
    white_max, black_max = calculate_max_nodes_per_color(moves_per_ply)
    assert white_max == 2 + 2 * 3 * 2
    assert black_max == 2 * 3


def test_calculate_min_nodes_per_color():
    white_min, black_min = calculate_min_nodes_per_color(5)
    assert white_min == 3
    assert black_min == 2

    white_min, black_min = calculate_min_nodes_per_color(4)
    assert white_min == 2
    assert black_min == 2


def test_calculate_color_sharpness_with_multiple_nodes():
    nodes = [
        TreeNode(cpl=10, lft=1, rgt=2, dfrc_id=1, cfg_id="1", move="e2e4"),
        TreeNode(cpl=30, lft=3, rgt=4, dfrc_id=1, cfg_id="1", move="g1f3"),
    ]
    max_nodes = 3
    min_nodes = 1
    result = calculate_color_sharpness(nodes, min_nodes, max_nodes)
    assert 0 < result < 1.0, f"Expected result to be between 0 and 1.0, got {result}"


def test_calculate_color_sharpness_with_no_nodes():
    nodes = []
    max_nodes = 3
    min_nodes = 1
    assert calculate_color_sharpness(nodes, min_nodes, max_nodes) is None


def test_calculate_color_sharpness_with_single_node():
    nodes = [
        TreeNode(cpl=10, lft=1, rgt=2, dfrc_id=1, cfg_id="1", move="e2e4"),
    ]
    max_nodes = 3
    min_nodes = 1
    assert calculate_color_sharpness(nodes, min_nodes, max_nodes) == 1.0


def test_sharpness_all_balanced():
    # Configuration with depth 4 and top moves count for both white and black
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=4,
        stockfish_depth_per_ply=[1, 1, 1, 1],
        num_top_moves_per_ply=[2, 2, 2, 2],
        balanced_threshold=10,
    )

    # Generate balanced nodes for a full binary tree
    nodes = []
    for i in range(config.analysis_depth_ply):  # Including root level at depth 0
        num_nodes_at_depth = config.num_top_moves_per_ply[i] ** i + 1

        nodes.extend(
            TreeNode(
                dfrc_id=1,
                cfg_id="cfg",
                lft=i,
                rgt=-1,
                move="e2e4",
                cpl=config.balanced_threshold - 1,
            )
            for _ in range(num_nodes_at_depth)
        )

    sharpness = calculate_position_sharpness(nodes, config)

    assert sharpness == Sharpness(white=0.0, black=0.0, total=0.0), (
        f"Expected sharpness to be 0.0 for all colors, got {sharpness}"
    )


def test_sharpness_none_below_threshold():
    # Configuration with depth 4
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=4,
        stockfish_depth_per_ply=[1, 1, 1, 1],
        num_top_moves_per_ply=[2, 2, 2, 2],
        balanced_threshold=10,
    )

    # No nodes within threshold (all moves are unbalanced)
    nodes = [
        TreeNode(dfrc_id=1, cfg_id="cfg", lft=i, rgt=i + 1, move="e2e4", cpl=20)
        for i in range(1, 5)
    ]

    sharpness = calculate_position_sharpness(nodes, config)

    assert sharpness == Sharpness(white=1.0, black=1.0, total=1.0)


def test_edge_case_no_nodes():
    # Configuration with depth 4
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=4,
        stockfish_depth_per_ply=[1, 1, 1, 1],
        num_top_moves_per_ply=[2, 2, 2, 2],
        balanced_threshold=10,
    )

    # Empty list of nodes
    nodes = []

    sharpness = calculate_position_sharpness(nodes, config)

    assert sharpness == Sharpness(white=None, black=None, total=None)


def test_non_linear_sharpness_scaling():
    # Configuration with depth 4
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=4,
        stockfish_depth_per_ply=[1, 1, 1, 1],
        num_top_moves_per_ply=[2, 2, 2, 2],
        balanced_threshold=10,
    )

    # Balanced nodes below threshold, but not all of them (partial sharpness)
    nodes = [
        TreeNode(
            dfrc_id=1,
            cfg_id="cfg",
            lft=i,
            rgt=i + 1,
            move="e2e4",
            cpl=5 if i % 2 == 0 else 15,
        )
        for i in range(1, 5)
    ]

    sharpness = calculate_position_sharpness(nodes, config)

    # We expect a non-linear scaling since only half moves are balanced
    expected_white_sharpness = 0.5  # Example value, depends on calculated ratio
    expected_black_sharpness = 1.0
    expected_total_sharpness = harmonic_mean(
        expected_white_sharpness,
        expected_black_sharpness,
    )

    assert sharpness == Sharpness(
        white=expected_white_sharpness,
        black=expected_black_sharpness,
        total=expected_total_sharpness,
    )


@pytest.mark.parametrize(
    "depth, moves, expected_white_min, expected_black_min",
    [
        (3, [2, 3, 1], 2, 1),
        (4, [3, 2, 4, 1], 3, 2),
        (5, [3, 3, 3, 3, 3], 3, 2),
    ],
)
def test_min_nodes_per_color_calculation(
    depth,
    moves,
    expected_white_min,
    expected_black_min,
):
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=depth,
        stockfish_depth_per_ply=[1] * depth,
        num_top_moves_per_ply=moves,
        balanced_threshold=10,
    )

    white_min, black_min = calculate_min_nodes_per_color(config.analysis_depth_ply)

    assert white_min == expected_white_min
    assert black_min == expected_black_min


@pytest.mark.parametrize(
    "depth, moves, expected_white_max, expected_black_max",
    [
        (3, [2, 3, 1], 6, 3),
        (4, [3, 2, 4, 1], 24, 12),
        (5, [3, 3, 3, 3, 3], 81, 81),
    ],
)
def test_max_nodes_per_color_calculation(
    depth,
    moves,
    expected_white_max,
    expected_black_max,
):
    config = AnalysisConfig(
        stockfish_version="Test",
        analysis_depth_ply=depth,
        stockfish_depth_per_ply=[1] * depth,
        num_top_moves_per_ply=moves,
        balanced_threshold=10,
    )

    white_max, black_max = calculate_max_nodes_per_color(
        config.analysis_depth_ply,
        config.num_top_moves_per_ply,
    )

    assert white_max == expected_white_max
    assert black_max == expected_black_max
