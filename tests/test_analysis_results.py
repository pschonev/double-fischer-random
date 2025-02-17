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
# Test 1: An empty tree (a leaf) should result in None sharpness for both sides.
###############################################################################
def test_sharpness_empty_tree():
    # A PositionNode with no children represents a leaf.
    root = make_node("start", 0, children=[])
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Since no moves are analyzed, both accumulators remain untouched and the score is None.
    assert (
        sharp.white is None
    ), f"Expected white sharpness to be None when no moves are present, got {sharp.white}"
    assert (
        sharp.black is None
    ), f"Expected black sharpness to be None when no moves are present, got {sharp.black}"
    assert (
        sharp.total is None
    ), f"Expected combined sharpness to be None when no moves are present, got {sharp.total}"


###############################################################################
# Test 2: A one-level tree where all moves are balanced.
#
# Note that the root analysis is from white’s perspective so the children (black moves)
# update the black accumulator. Since white made no decision, its accumulator remains empty.
###############################################################################
def test_sharpness_all_balanced():
    # Create a root that has 5 children (with cpl=10 <= BALANCED_THRESHOLD => balanced).
    children = [make_node(f"move{i}", 10) for i in range(5)]
    root = make_node("start", 0, children=children)
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Expected outcome:
    #   white sharpness remains None (no decision made by white),
    #   black sharpness becomes 0.0 (all available moves balanced, so minimal forcing),
    #   combined sharpness is None (since one branch is missing).
    assert (
        sharp.white is None
    ), f"Expected white sharpness to be None when no decision for white, got {sharp.white}"
    assert sharp.black is not None, "Expected black sharpness to be non-None."
    assert isclose(sharp.black, 0.0), f"Expected black sharpness 0.0, got {sharp.black}"
    assert (
        sharp.total is None
    ), f"Expected combined sharpness to be None when one side is None, got {sharp.total}"


###############################################################################
# Test 3: A one-level tree where only one out of 5 moves is balanced.
#
# Here, only the opponent’s accumulator gets updated, so for a tree starting from white,
# the black accumulator gets one balanced move among five. By design, a singleton is maximally forcing.
###############################################################################
def test_sharpness_one_balanced_among_unbalanced():
    balanced_child = make_node("balanced_move", 10)  # Good move.
    unbalanced_children = [make_node(f"unbalanced_move{i}", 30) for i in range(4)]
    children = [balanced_child] + unbalanced_children
    root = make_node("start", 0, children=children)
    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Expected outcome:
    #   white sharpness remains None (no decision by white),
    #   black sharpness becomes 1.0 (only one balanced move available at this level),
    #   combined sharpness is None due to the missing white branch.
    assert (
        sharp.white is None
    ), f"Expected white sharpness to be None when no decision for white, got {sharp.white}"
    assert sharp.black is not None, "Expected black sharpness to be non-None."
    assert isclose(
        sharp.black, 1.0, rel_tol=1e-6
    ), f"Expected black sharpness 1.0, got {sharp.black}"
    assert (
        sharp.total is None
    ), f"Expected combined sharpness to be None when one branch is None, got {sharp.total}"


###############################################################################
# Test 4: Extended branch where one branch extends to further depths.
#
# Tree structure:
#  - Root has 5 children (black moves). Only one child (A) is balanced.
#  - Node A (a black move) has 5 children (white moves), only one (A1) is balanced.
#  - Node A1 (a white move) has 5 children (black moves), only one balanced.
#
# Accumulators:
#  • Level 1: black_acc becomes (balanced_moves = 1, total = 5) → returns 1.0.
#  • Level 2: white_acc becomes (balanced_moves = 1, total = 5) → returns 1.0.
#  • Level 3: black_acc gets additional (balanced_moves = 1, total = 5), so overall black_acc is (2, 10)
#     yielding 1.0 - ((2-1)/(10-1)) = 8/9 ≈ 0.888889.
# Combined sharpness is the harmonic mean of (1.0, 8/9) = 16/17 ≈ 0.941176.
###############################################################################
def test_sharpness_extended_branch():
    # Level 1:
    unbalanced_level1 = [make_node(f"unbalanced{i}", 30) for i in range(1, 5)]
    child_A = make_node("A", 10, children=[])  # Balanced child to be extended.
    root_children = [child_A] + unbalanced_level1
    root = make_node("start", 0, children=root_children)

    # Level 2 (children of A; moves for white):
    balanced_A1 = make_node("A1", 10, children=[])  # Only one balanced move.
    unbalanced_A_children = [make_node(f"A_unbal{i}", 30) for i in range(1, 5)]
    level2_children = [balanced_A1] + unbalanced_A_children
    child_A.children = level2_children

    # Level 3 (children of A1; moves for black):
    balanced_A1_child = make_node("A1a", 10, children=[])
    unbalanced_A1_children = [make_node(f"A1_unbal{i}", 30) for i in range(1, 5)]
    level3_children = [balanced_A1_child] + unbalanced_A1_children
    balanced_A1.children = level3_children

    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    expected_black = 8 / 9  # ≈ 0.888889
    expected_white = 1.0
    # Harmonic mean of 1.0 and 8/9: (2 * 1.0 * (8/9)) / (1.0 + (8/9)) = 16/17 ≈ 0.941176
    expected_total = (2 * expected_white * expected_black) / (
        expected_white + expected_black
    )
    # Here both accumulators contribute, so neither is None.
    assert sharp.white is not None, "Expected white sharpness to be non-None."
    assert sharp.black is not None, "Expected black sharpness to be non-None."
    assert sharp.total is not None, "Expected combined sharpness to be non-None."
    assert isclose(
        sharp.white, expected_white, rel_tol=1e-6
    ), f"Expected white sharpness {expected_white}, got {sharp.white}"
    assert isclose(
        sharp.black, expected_black, rel_tol=1e-6
    ), f"Expected black sharpness {expected_black}, got {sharp.black}"
    assert isclose(
        sharp.total, expected_total, rel_tol=1e-6
    ), f"Expected combined sharpness {expected_total}, got {sharp.total}"


###############################################################################
# Test 5: Extended branch where at one level there are 0 balanced moves.
#
# In this branch:
#  - Level 1 (white True): black_acc gets (balanced_moves = 1, total = 5) → returns 1.0.
#  - Level 2 (child A's children, white False): white_acc gets (balanced_moves = 0, total = 5)
#    and therefore yields None.
# The overall combined sharpness becomes None.
###############################################################################
def test_sharpness_zero_balanced_extended_branch():
    # Level 1:
    unbalanced_level1 = [make_node(f"unbal1_{i}", 30) for i in range(1, 5)]
    child_A = make_node("A", 10, children=[])  # Balanced initially to allow recursion.
    root_children = [child_A] + unbalanced_level1
    root = make_node("start", 0, children=root_children)

    # Level 2 (children of A; moves for white):
    level2_children = [make_node(f"A_level2_{i}", 30, children=[]) for i in range(5)]
    child_A.children = level2_children

    sharp = AnalysisResult._calculate_sharpness_score(root, BALANCED_THRESHOLD)
    # Expected:
    #   white sharpness is None (no balanced moves found at level 2),
    #   black sharpness remains 1.0 (from level 1),
    #   combined sharpness is None.
    assert (
        sharp.white is None
    ), f"Expected white sharpness to be None when no balanced moves are available, got {sharp.white}"
    assert sharp.black is not None, "Expected black sharpness to be non-None."
    assert isclose(
        sharp.black, 1.0, rel_tol=1e-6
    ), f"Expected black sharpness 1.0, got {sharp.black}"
    assert (
        sharp.total is None
    ), f"Expected combined sharpness to be None due to one missing branch, got {sharp.total}"
