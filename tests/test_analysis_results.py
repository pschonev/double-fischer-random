from math import isclose
from src.analysis_results import AnalysisResult, PositionNode


# Helper to build PositionNode instances.
def make_node(
    move: str, cpl: int, children: list | None = None, pv: list | None = None
):
    if children is None:
        children = []
    if pv is None:
        pv = []
    return PositionNode(move=move, cpl=cpl, children=children, pv=pv)


# We will use this threshold in our tests.
BALANCED_THRESHOLD = 20


###############################################################################
# Test 1: An empty tree (a leaf) should result in 0 sharpness for both sides.
###############################################################################
def test_sharpness_empty_tree():
    # A PositionNode with no children represents a leaf.
    root = make_node("start", 0, children=[])
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Since no moves were analyzed, both accumulators remain untouched.
    assert sharp.white == 0.0, "Expected white sharpness 0 when no moves are present"
    assert sharp.black == 0.0, "Expected black sharpness 0 when no moves are present"
    assert sharp.total == 0.0, "Expected combined sharpness 0 when no moves are present"


###############################################################################
# Test 2: A one-level tree where all moves are balanced.
###############################################################################
def test_sharpness_all_balanced():
    # Create a root that has 5 children.
    # (At the root, children represent moves for black.)
    children = [
        make_node(f"move{i}", 10) for i in range(5)
    ]  # cpl=10 <= BALANCED_THRESHOLD => balanced
    root = make_node("start", 0, children=children)
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Here black_acc gets balanced_moves = 5 and total = 5,
    # so sharpness = 1 - ((5 - 1)/(5 - 1)) = 0.
    assert isclose(sharp.black, 0.0), f"Expected black sharpness 0, got {sharp.black}"
    assert isclose(sharp.white, 0.0), f"Expected white sharpness 0, got {sharp.white}"
    assert isclose(
        sharp.total, 0.0
    ), f"Expected combined sharpness 0, got {sharp.total}"


###############################################################################
# Test 3: A one-level tree where only one out of 5 moves is balanced.
#
# Previously one balanced among 5 yielded 0.8; now by design, a single good move
# should give maximum (1.0) sharpness even if the total moves are 5.
# In this test, only black gets a decision (so white remains 0), and the harmonic
# mean of (1.0, 0) is 0.
###############################################################################
def test_sharpness_one_balanced_among_unbalanced():
    balanced_child = make_node("balanced_move", 10)  # Good move.
    unbalanced_children = [make_node(f"unbalanced_move{i}", 30) for i in range(4)]
    children = [balanced_child] + unbalanced_children
    root = make_node("start", 0, children=children)
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # At root: total = 5, balanced = 1, so (by our new rule) black_sharpness = 1.0.
    # There is no further recursion so white remains 0.
    assert isclose(
        sharp.black, 1.0, rel_tol=1e-6
    ), f"Expected black sharpness 1.0, got {sharp.black}"
    assert isclose(
        sharp.white, 0.0, rel_tol=1e-6
    ), f"Expected white sharpness 0, got {sharp.white}"
    assert isclose(
        sharp.total, 0.0, rel_tol=1e-6
    ), f"Expected combined sharpness 0, got {sharp.total}"


###############################################################################
# Test 4: Extended branch where only one branch is extended to further depths.
#
# Tree structure:
#  - Root has 5 children (black moves). Only one child (A) is balanced.
#  - Node A (a black move) has 5 children (white moves), only one (A1) is balanced.
#  - Node A1 (a white move) has 5 children (black moves), only one balanced.
#
# Accumulators:
#  • Level 1 (root): black_acc gets (balanced_moves = 1, total = 5) → contributes 1.0 (since 1 good move is max).
#  • Level 2 (child A, white move): white_acc gets (balanced_moves = 1, total = 5) → becomes 1.0.
#  • Level 3 (child A1, black move): black_acc gets additional (balanced_moves = 1, total = 5).
#     Thus overall, black_acc: balanced_moves=2, total=10.
#     Score = 1 - ((2-1)/(10-1)) = 1 - (1/9) = 8/9 (≈ 0.888889)
#
# The combined sharpness, computed via harmonic mean of (8/9, 1.0),
# is: (2 * (8/9) * 1.0) / ((8/9) + 1.0) = 16/17 (≈ 0.941176).
###############################################################################
def test_sharpness_extended_branch():
    # Level 1:
    unbalanced_level1 = [make_node(f"unbalanced{i}", 30) for i in range(1, 5)]
    child_A = make_node("A", 10, children=[])  # balanced child to be extended.
    root_children = [child_A] + unbalanced_level1
    root = make_node("start", 0, children=root_children)

    # Level 2 (children of A; moves for white):
    balanced_A1 = make_node("A1", 10, children=[])  # balanced move at level 2.
    unbalanced_A_children = [make_node(f"A_unbal{i}", 30) for i in range(1, 5)]
    level2_children = [balanced_A1] + unbalanced_A_children
    child_A.children = level2_children

    # Level 3 (children of A1; moves for black):
    balanced_A1_child = make_node("A1a", 10, children=[])
    unbalanced_A1_children = [make_node(f"A1_unbal{i}", 30) for i in range(1, 5)]
    level3_children = [balanced_A1_child] + unbalanced_A1_children
    balanced_A1.children = level3_children

    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Expected values:
    #   black_sharpness = 8/9 ≈ 0.888889,
    #   white_sharpness = 1.0,
    #   combined sharpness = 16/17 ≈ 0.941176.
    assert isclose(
        sharp.black, 8 / 9, rel_tol=1e-6
    ), f"Expected black sharpness {8/9}, got {sharp.black}"
    assert isclose(
        sharp.white, 1.0, rel_tol=1e-6
    ), f"Expected white sharpness 1.0, got {sharp.white}"
    assert isclose(
        sharp.total, 16 / 17, rel_tol=1e-6
    ), f"Expected combined sharpness {16/17}, got {sharp.total}"


###############################################################################
# Test 5: Extended branch where at a level there are 0 balanced moves.
#
# In this branch:
#  - Level 1 (with white True): black_acc gets (balanced_moves = 1, total = 5) → score becomes 1.0.
#  - Level 2 (child A's children, with white False): white_acc gets (balanced_moves = 0, total = 5)
#    and therefore yields sharpness = 0.
# The overall combined sharpness is the harmonic mean of (1.0, 0.0) which is 0.
###############################################################################
def test_sharpness_zero_balanced_extended_branch():
    # Level 1:
    unbalanced_level1 = [make_node(f"unbal1_{i}", 30) for i in range(1, 5)]
    child_A = make_node("A", 10, children=[])  # balanced to allow recursion.
    root_children = [child_A] + unbalanced_level1
    root = make_node("start", 0, children=root_children)

    # Level 2 (children of A; moves for white):
    level2_children = [make_node(f"A_level2_{i}", 30, children=[]) for i in range(5)]
    child_A.children = level2_children

    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Now, expected:
    #   white_sharpness should be 0 (no balanced moves),
    #   black_sharpness is 1.0 (from level 1, one good move among 5),
    #   combined, via harmonic mean, should be 0.
    assert isclose(
        sharp.white, 0.0, abs_tol=1e-6
    ), f"Expected white sharpness 0 when no good moves are available, got {sharp.white}"
    assert isclose(
        sharp.black, 1.0, rel_tol=1e-6
    ), f"Expected black sharpness 1.0, got {sharp.black}"
    assert isclose(
        sharp.total, 0.0, abs_tol=1e-6
    ), f"Expected combined sharpness 0, got {sharp.total}"
