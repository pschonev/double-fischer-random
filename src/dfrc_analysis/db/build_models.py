from dfrc_analysis.analysis.eval import Sharpness, calculate_balance_score
from dfrc_analysis.analysis.results import (
    AnalysisData,
    AnalysisParams,
    PositionNode,
)
from dfrc_analysis.db.models import AnalysisResult, TreeNode
from dfrc_analysis.positions.positions import (
    chess960_to_dfrc_uid,
    get_chess960_position,
    is_flipped,
    is_mirrored,
)
from dfrc_analysis.utils import harmonic_mean

#
# Tree
#

AnalysisTree = PositionNode


def _convert_tree_to_nested_set(
    node: PositionNode,
    dfrc_id: int,
    cfg_id: str,
    left: int = 1,
) -> tuple[list[TreeNode], int]:
    """
    Convert a tree structure to a list of TreeNode objects using the nested set model.

    Args:
        node: The current node in the tree
        dfrc_id: The dfrc_id of the associated analysis result
        cfg_id: The cfg_id of the associated analysis result
        left: The left value for the current node

    Returns:
        A tuple containing the list of TreeNode objects and the right value for the current node
    """
    nodes = []
    right = left + 1

    # Process all children
    for child in node.children:
        child_nodes, child_right = _convert_tree_to_nested_set(
            child,
            dfrc_id,
            cfg_id,
            right,
        )
        nodes.extend(child_nodes)
        right = child_right + 1

    # Create TreeNode for the current node
    current_node = TreeNode(
        dfrc_id=dfrc_id,
        cfg_id=cfg_id,
        lft=left,
        rgt=right,
        move=node.move
        if hasattr(node, "move")
        else "",  # Root node might not have a move
        cpl=node.analysis.cpl,
        mate=node.analysis.mate,
        pv=node.analysis.pv,
    )

    nodes.append(current_node)
    return nodes, right


def convert_analysis_tree(
    data: AnalysisParams,
    analysis_tree: AnalysisTree,
) -> list[TreeNode]:
    """Convert the analysis tree to a list of TreeNode objects"""
    dfrc_id = chess960_to_dfrc_uid(data.white_id, data.black_id)
    tree_nodes, _ = _convert_tree_to_nested_set(analysis_tree, dfrc_id, data.cfg_id)
    return tree_nodes


#
# Analysis Result
#


def build_analysis_result(
    data: AnalysisData,
    sharpness: Sharpness,
) -> AnalysisResult:
    """Build an AnalysisResult using the provided data and pre-calculated sharpness"""
    white = get_chess960_position(data.params.white_id)
    black = get_chess960_position(data.params.black_id)

    root_analysis = data.analysis_tree.analysis
    balance_score = calculate_balance_score(root_analysis.cpl, root_analysis.mate)

    playability_score = None
    if sharpness.total is not None:
        playability_score = harmonic_mean(
            balance_score,
            sharpness.total,
        )

    return AnalysisResult(
        dfrc_id=chess960_to_dfrc_uid(
            white=data.params.white_id,
            black=data.params.black_id,
        ),
        white_id=data.params.white_id,
        black_id=data.params.black_id,
        white=white,
        black=black,
        cfg_id=data.params.cfg_id,
        threads=data.params.threads,
        hash_size=data.params.hash,
        analyzer=data.analyzer,
        starting_pos_cpl=root_analysis.cpl,
        starting_pos_mate=root_analysis.mate,
        playability_score=playability_score,
        white_sharpness=sharpness.white,
        black_sharpness=sharpness.black,
        total_sharpness=sharpness.total,
        balance_score=balance_score,
        mirrored=is_mirrored(white, black),
        flipped=is_flipped(white, black),
        swapped_id=chess960_to_dfrc_uid(
            white=data.params.black_id,
            black=data.params.white_id,
        ),
    )
