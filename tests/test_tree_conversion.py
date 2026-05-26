import pytest
from dfrc_analysis.analysis.results import PositionNode, PositionAnalysis
from dfrc_analysis.db.build_models import _convert_tree_to_nested_set
from dfrc_analysis.db.models import TreeNode


def convert_nested_set_to_tree(nodes: list[TreeNode]) -> PositionNode:
    """Convert a list of TreeNode objects back to a tree structure."""
    # Create position nodes
    position_nodes = {}
    for node in nodes:
        position_node = PositionNode(
            move=node.move,
            children=[],
            analysis=PositionAnalysis(cpl=node.cpl, mate=node.mate, pv=node.pv),
        )
        position_nodes[(node.lft, node.rgt)] = position_node

    # Build the tree
    for node in nodes:
        # Find the direct parent
        parent = None
        min_rgt = float("inf")
        for potential_parent in nodes:
            if (
                potential_parent.lft < node.lft
                and potential_parent.rgt > node.rgt
                and potential_parent.rgt < min_rgt
            ):
                parent = potential_parent
                min_rgt = potential_parent.rgt

        if parent:
            position_nodes[(parent.lft, parent.rgt)].children.append(
                position_nodes[(node.lft, node.rgt)]
            )

    # Find the root node
    root_node = next(
        node
        for node in nodes
        if not any(
            potential_parent.lft < node.lft and potential_parent.rgt > node.rgt
            for potential_parent in nodes
        )
    )

    return position_nodes[(root_node.lft, root_node.rgt)]


def test_conversion_roundtrip():
    """Test conversion from tree to nested set and back."""
    # Create a sample tree
    leaf1 = PositionNode(
        move="leaf1",
        children=[],
        analysis=PositionAnalysis(cpl=10, mate=None, pv=["pv1"]),
    )
    leaf2 = PositionNode(
        move="leaf2",
        children=[],
        analysis=PositionAnalysis(cpl=20, mate=None, pv=["pv2"]),
    )
    child1 = PositionNode(
        move="child1",
        children=[leaf1, leaf2],
        analysis=PositionAnalysis(cpl=5, mate=None, pv=["pv3"]),
    )

    leaf3 = PositionNode(
        move="leaf3",
        children=[],
        analysis=PositionAnalysis(cpl=30, mate=None, pv=["pv4"]),
    )
    leaf4 = PositionNode(
        move="leaf4",
        children=[],
        analysis=PositionAnalysis(cpl=40, mate=None, pv=["pv5"]),
    )
    child2 = PositionNode(
        move="child2",
        children=[leaf3, leaf4],
        analysis=PositionAnalysis(cpl=15, mate=None, pv=["pv6"]),
    )

    root = PositionNode(
        move="root",
        children=[child1, child2],
        analysis=PositionAnalysis(cpl=0, mate=None, pv=["pv7"]),
    )

    # Convert to nested set
    nodes, _ = _convert_tree_to_nested_set(root, 1, "test", 1)

    # Convert back to tree
    reconstructed = convert_nested_set_to_tree(nodes)

    # Verify the structure is preserved
    assert reconstructed.move == root.move
    assert len(reconstructed.children) == len(root.children)

    # Check first branch
    assert reconstructed.children[0].move == child1.move
    assert len(reconstructed.children[0].children) == 2
    assert {c.move for c in reconstructed.children[0].children} == {"leaf1", "leaf2"}

    # Check second branch
    assert reconstructed.children[1].move == child2.move
    assert len(reconstructed.children[1].children) == 2
    assert {c.move for c in reconstructed.children[1].children} == {"leaf3", "leaf4"}


def test_node_count_by_level():
    """Test that each level has the expected number of nodes."""

    # Create a tree with uniform branching factor of 5
    def create_tree_with_branching(branching_factor, depth):
        if depth == 0:
            return PositionNode(
                move="leaf",
                children=[],
                analysis=PositionAnalysis(cpl=0, mate=None, pv=None),
            )

        children = []
        for _ in range(branching_factor):
            children.append(create_tree_with_branching(branching_factor, depth - 1))

        return PositionNode(
            move=f"node_depth_{depth}",
            children=children,
            analysis=PositionAnalysis(cpl=0, mate=None, pv=None),
        )

    # Create a tree with branching factor 5 and depth 2
    root = create_tree_with_branching(5, 2)

    # Convert to nested set
    nodes, _ = _convert_tree_to_nested_set(root, 1, "test", 1)

    # Count nodes at each level
    level_counts = {}
    for node in nodes:
        # Calculate level based on containing nodes
        level = sum(
            1 for other in nodes if other.lft < node.lft and other.rgt > node.rgt
        )
        level_counts[level] = level_counts.get(level, 0) + 1

    # Verify counts
    assert level_counts[0] == 1  # Root
    assert level_counts[1] == 5  # First level (5^1)
    assert level_counts[2] == 25  # Second level (5^2)


def test_lft_odd_even_pattern():
    """Test that lft values follow an odd/even pattern based on tree level."""
    # Create a sample tree
    leaf1 = PositionNode(
        move="leaf1",
        children=[],
        analysis=PositionAnalysis(cpl=10, mate=None, pv=["pv1"]),
    )
    leaf2 = PositionNode(
        move="leaf2",
        children=[],
        analysis=PositionAnalysis(cpl=20, mate=None, pv=["pv2"]),
    )
    child1 = PositionNode(
        move="child1",
        children=[leaf1, leaf2],
        analysis=PositionAnalysis(cpl=5, mate=None, pv=["pv3"]),
    )

    leaf3 = PositionNode(
        move="leaf3",
        children=[],
        analysis=PositionAnalysis(cpl=30, mate=None, pv=["pv4"]),
    )
    child2 = PositionNode(
        move="child2",
        children=[leaf3],
        analysis=PositionAnalysis(cpl=15, mate=None, pv=["pv6"]),
    )

    root = PositionNode(
        move="root",
        children=[child1, child2],
        analysis=PositionAnalysis(cpl=0, mate=None, pv=["pv7"]),
    )

    # Convert to nested set
    nodes, _ = _convert_tree_to_nested_set(root, 1, "test", 1)

    # Group nodes by level
    level_to_nodes = {}
    for node in nodes:
        level = sum(
            1 for other in nodes if other.lft < node.lft and other.rgt > node.rgt
        )
        if level not in level_to_nodes:
            level_to_nodes[level] = []
        level_to_nodes[level].append(node)

    # Check that nodes at same level have consistent odd/even pattern
    for level, level_nodes in level_to_nodes.items():
        is_odd = level_nodes[0].lft % 2 == 1
        for node in level_nodes:
            assert (node.lft % 2 == 1) == is_odd

    # Check that adjacent levels have opposite patterns
    if len(level_to_nodes) > 1:
        for level in range(max(level_to_nodes.keys())):
            if level in level_to_nodes and level + 1 in level_to_nodes:
                is_odd_current = level_to_nodes[level][0].lft % 2 == 1
                is_odd_next = level_to_nodes[level + 1][0].lft % 2 == 1
                assert is_odd_current != is_odd_next


def test_sample_data():
    """Test with the provided sample data."""
    # Create sample data
    sample_data = [
        TreeNode(
            dfrc_id=520750,
            cfg_id="XS",
            lft=1,
            rgt=2,
            move="root",
            cpl=51,
            mate=None,
            pv=[
                "d2d4",
                "d7d5",
                "f2f3",
                "b7b6",
                "b1c3",
                "c7c6",
                "a2a4",
                "e7e5",
                "d4e5",
                "b8e5",
                "e2e4",
                "h7h5",
                "a4a5",
                "a8b8",
                "e4d5",
                "h8h6",
                "e1d3",
                "e5c3",
                "b2c3",
                "g8h7",
                "d1d2",
                "h6d6",
                "d3f4",
                "c8e7",
                "g1d4",
                "e7d5",
                "f4d5",
                "c6d5",
                "a5b6",
                "a7b6",
                "f1b5",
                "d8c6",
                "h1e1",
                "e8e1",
                "d2e1",
                "c6d4",
                "c3d4",
            ],
        ),
        TreeNode(
            dfrc_id=536094,
            cfg_id="XS",
            lft=4,
            rgt=5,
            move="d7d5",
            cpl=-10,
            mate=None,
            pv=[
                "d7d5",
                "f2f3",
                "c7c6",
                "e2e4",
                "e7e5",
                "d4e5",
                "b8e5",
                "b1c3",
                "a8b8",
                "e1d2",
                "d5e4",
                "g1c5",
                "c8e7",
                "c3e4",
                "d8e6",
                "c5d6",
                "e5d6",
                "d2d6",
                "f6f5",
                "d6b8",
                "e8b8",
                "e4f2",
                "b8d8",
                "a1a3",
                "e6d4",
                "a3d3",
                "h7h5",
                "h2h4",
                "c6c5",
                "d3d2",
            ],
        ),
        TreeNode(
            dfrc_id=536094,
            cfg_id="XS",
            lft=6,
            rgt=7,
            move="d7d5",
            cpl=-3,
            mate=None,
            pv=[
                "d7d5",
                "f2f3",
                "c7c6",
                "b1c3",
                "e7e5",
                "d4e5",
                "b8e5",
                "e4d5",
                "c6d5",
                "e1d2",
                "d8c6",
                "g1c5",
                "c8e7",
                "f3f4",
                "e5b8",
                "g2g3",
                "d5d4",
                "c3e4",
                "b8c7",
            ],
        ),
        TreeNode(
            dfrc_id=536094,
            cfg_id="XS",
            lft=3,
            rgt=14,
            move="a2a4",
            cpl=-12,
            mate=None,
            pv=None,
        ),
    ]

    # Verify nested set properties
    for node in sample_data:
        assert node.rgt > node.lft  # Right value always greater than left

    # Convert to tree
    tree = convert_nested_set_to_tree(sample_data)

    # Verify structure
    assert tree.move == "root"
    assert len(tree.children) == 1
    assert tree.children[0].move == "a2a4"

    # Convert back to nested set
    reconstructed_nodes, _ = _convert_tree_to_nested_set(tree, 520750, "XS", 1)

    # Verify structure is maintained
    reconstructed_tree = convert_nested_set_to_tree(reconstructed_nodes)
    assert reconstructed_tree.move == "root"
    assert len(reconstructed_tree.children) == 1
    assert reconstructed_tree.children[0].move == "a2a4"


def test_large_tree_performance():
    """Test performance with a large tree."""

    # Create a large tree with branching factor 3 and depth 5
    def create_large_tree(branching_factor, depth, prefix=""):
        if depth == 0:
            return PositionNode(
                move=f"{prefix}leaf",
                children=[],
                analysis=PositionAnalysis(cpl=0, mate=None, pv=None),
            )

        children = []
        for i in range(branching_factor):
            children.append(
                create_large_tree(branching_factor, depth - 1, f"{prefix}{i}_")
            )

        return PositionNode(
            move=f"{prefix}node_depth_{depth}",
            children=children,
            analysis=PositionAnalysis(cpl=0, mate=None, pv=None),
        )

    # Create a tree with branching factor 3 and depth 5
    # This will have 3^0 + 3^1 + 3^2 + 3^3 + 3^4 + 3^5 = 1 + 3 + 9 + 27 + 81 + 243 = 364 nodes
    root = create_large_tree(3, 5)

    # Convert to nested set
    nodes, right = _convert_tree_to_nested_set(root, 1, "test", 1)

    # Verify node count
    assert len(nodes) == 364

    # Verify right value
    assert right == 729  # 2 * number of nodes + 1
