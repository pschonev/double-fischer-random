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
    count_nodes_by_color,
    filter_balanced_nodes,
)


@pytest.mark.parametrize(
    "a, b, power, expected",
    [
        (0, 10, 1, 1.0),
        (10, 10, 1, 0.0),
        (5, 10, 1, 0.5),
        (5, 10, 2, 0.75),
        (8, 20, 1, 0.6),
        (12, 20, 1, 0.4),
    ],
)
def test_calculate_sharpness_ratio(a, b, power, expected):
    assert calculate_sharpness_ratio(a, b, power=power) == expected


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
    white_node_count, black_node_count = count_nodes_by_color(nodes)
    assert white_node_count == 2
    assert black_node_count == 2


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
    result = calculate_color_sharpness(len(nodes), min_nodes, max_nodes)
    assert result is not None
    assert 0 < result < 1.0, f"Expected result to be between 0 and 1.0, got {result}"


def test_calculate_color_sharpness_with_single_node():
    nodes = 1
    max_nodes = 3
    min_nodes = 1
    assert calculate_color_sharpness(nodes, min_nodes, max_nodes) == 1.0


class FullTreeTestCase:
    @pytest.fixture
    def config(self):
        # Configuration with depth 4 and top moves count for both white and black
        return AnalysisConfig(
            stockfish_version="Test",
            analysis_depth_ply=4,
            stockfish_depth_per_ply=[1, 1, 1, 1],
            num_top_moves_per_ply=[2, 2, 2, 2],
            balanced_threshold=10,
        )

    @pytest.fixture
    def nodes(self, config):
        # Generate balanced nodes for a full binary tree
        nodes = []
        num_nodes_at_depth = 1  # Start with 1 for the root node
        for i in range(config.analysis_depth_ply):  # Including root level at depth 0
            num_nodes_at_depth = config.num_top_moves_per_ply[i] * num_nodes_at_depth
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
        return nodes

    def test_balanced_nodes_count(self, config, nodes):
        assert len(filter_balanced_nodes(nodes, config.balanced_threshold)) == 30

    def test_nodes_color_count(self, nodes):
        assert count_nodes_by_color(nodes) == (10, 20)

    def test_max_nodes_per_color(self, config):
        assert calculate_max_nodes_per_color(config.num_top_moves_per_ply) == (10, 20)

    def test_min_nodes_per_color(self, config):
        assert calculate_min_nodes_per_color(config.analysis_depth_ply) == (2, 2)

    def test_white_color_sharpness(self):
        assert calculate_color_sharpness(10, 2, 10) == 0.0  # white sharpness

    def test_black_color_sharpness(self):
        assert calculate_color_sharpness(20, 2, 20) == 0.0  # black sharpness

    def test_calculated_sharpness(self, config, nodes):
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

    assert sharpness == Sharpness(white=None, black=None, total=None)


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


@pytest.mark.parametrize(
    "depth, moves, expected_white_min, expected_black_min",
    [
        (3, [2, 3, 1], 2, 1),
        (4, [3, 2, 4, 1], 2, 2),
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
    "moves, expected_white_max, expected_black_max",
    [
        ([2, 3, 1], 8, 6),
        ([3, 2, 4, 1], 27, 30),
        ([3, 3, 3, 3, 3], 273, 90),
    ],
)
def test_max_nodes_per_color_calculation(
    moves,
    expected_white_max,
    expected_black_max,
):
    white_max, black_max = calculate_max_nodes_per_color(
        moves,
    )

    assert white_max == expected_white_max
    assert black_max == expected_black_max
