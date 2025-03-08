from dataclasses import dataclass
from datetime import datetime

import chess.engine
import msgspec
from sqlalchemy import ARRAY, Column, String
from sqlmodel import Field, PrimaryKeyConstraint, SQLModel

from src.analysis_config import AnalysisConfig, load_config
from src.positions import (
    chess960_to_dfrc_uid,
    get_chess960_position,
    is_flipped,
    is_mirrored,
)
from src.utils import harmonic_mean


class PositionAnalysis(msgspec.Struct):
    cpl: int | None
    mate: int | None
    pv: list[str] | None


class PositionNode(msgspec.Struct):
    """Node in the position analysis tree which represents a halfmove (ply) and its evaluation"""

    move: str
    children: list["PositionNode"]
    analysis: PositionAnalysis


@dataclass
class AnalysisParams:
    """Contains the data from an analysis"""

    white_id: int
    black_id: int

    threads: int
    hash: int

    cfg_id: str


class AnalysisData(msgspec.Struct):
    """Contains the data from an analysis"""

    params: AnalysisParams
    analyzer: str  # github user
    analysis_tree: PositionNode


class AnalysisResult(SQLModel, table=True):
    """
    SQLModel representation of analysis results.
    This is a flattened version of the nested msgspec structure.
    """

    __tablename__ = "analysis_results"  # type: ignore

    # Composite primary key
    dfrc_id: int = Field(primary_key=True)
    cfg_id: str = Field(primary_key=True)

    # Position details
    white_id: int
    black_id: int
    white: str
    black: str

    # Analysis metadata
    analyzer: str  # GitHub user
    threads: int
    hash_size: int  # renamed from 'hash' which is a reserved keyword

    # Root position analysis
    starting_pos_cpl: int | None = None
    starting_pos_mate: int | None = None

    # Analysis results
    white_sharpness: float | None = None
    black_sharpness: float | None = None
    total_sharpness: float | None = None
    playability_score: float | None = None

    # Position properties
    mirrored: bool
    flipped: bool

    # Timestamp for when the analysis was created
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        PrimaryKeyConstraint("dfrc_id", "cfg_id", name="analysis_result_pk"),
    )


class TreeNode(SQLModel, table=True):
    """
    Represents a node in the position analysis tree using the nested set model.
    Each node represents a halfmove (ply) and its evaluation.
    """

    __tablename__ = "tree_nodes"  # type: ignore

    # Composite primary key
    dfrc_id: int = Field(primary_key=True)
    cfg_id: str = Field(primary_key=True)
    lft: int = Field(primary_key=True)

    # Nested set model fields
    rgt: int = Field(index=True)

    # Node data
    move: str  # The chess move in UCI format

    # Analysis data
    cpl: int | None = None
    mate: int | None = None

    # PostgreSQL array for PV (principal variation)
    pv: list[str] | None = Field(default=None, sa_column=Column(ARRAY(String)))

    __table_args__ = (
        PrimaryKeyConstraint("dfrc_id", "cfg_id", "lft", name="treenode_pk"),
    )


def calculate_balance_score(cpl: int | None, mate: int | None) -> float:
    """Calculate the balance score from the win, draw, loss probabilities

    Args:
        cpl: Centipawn loss value
        mate: Mate in N moves value

    Returns:
        A float between 0 and 1 representing the balance of the position.
        0 is perfectly balanced, 1 is completely unbalanced.
    """
    if mate is None and cpl is None:
        raise ValueError("At least one of mate or cpl must be provided")
    if cpl is None:
        return 1
    wdl = chess.engine.Cp(cpl).wdl()
    return abs(wdl.wins - wdl.losses) / 1000


class Sharpness(msgspec.Struct):
    """
    Contains the sharpness of a position.
    A `None` value indicates that no good moves were available.
    """

    white: float | None
    black: float | None
    total: float | None


def calculate_sharpness_score(
    nodes: list[TreeNode],
    cfg: AnalysisConfig,
) -> Sharpness:
    # Calculate theoretical maximum nodes for white and black
    def calculate_max_possible_nodes(*, is_white: bool) -> int:
        total = 0
        current_product = 1

        # Determine which plies belong to this player
        start_ply = 0 if is_white else 1

        for ply in range(start_ply, cfg.analysis_depth_ply, 2):
            current_product *= cfg.num_top_moves_per_ply[ply]
            total += current_product

        return total

    white_max = calculate_max_possible_nodes(is_white=True)
    black_max = calculate_max_possible_nodes(is_white=False)

    # Count balanced nodes for white and black
    white_balanced = sum(
        1
        for node in nodes
        if node.cpl is not None
        and abs(node.cpl) <= cfg.balanced_threshold
        and (node.lft % 2 == 1)  # White's moves are at odd levels in the tree
    )

    black_balanced = sum(
        1
        for node in nodes
        if node.cpl is not None
        and abs(node.cpl) <= cfg.balanced_threshold
        and (node.lft % 2 == 0)  # Black's moves are at even levels in the tree
    )

    # Calculate minimum nodes needed for a single complete line
    white_min = (cfg.analysis_depth_ply + 1) // 2  # Ceiling division for white
    black_min = cfg.analysis_depth_ply // 2  # Floor division for black

    # Calculate sharpness scores
    def calculate_score(
        balanced: int,
        min_needed: int,
        max_possible: int,
    ) -> float | None:
        if balanced < min_needed:
            return None  # Not enough balanced nodes for even one complete line

        if balanced == min_needed:
            return 1.0  # Maximum sharpness - only one line is playable

        # Calculate how many nodes beyond the minimum we have
        extra_balanced = balanced - min_needed
        extra_possible = max_possible - min_needed

        # Return inverted ratio (1.0 = max sharpness, 0.0 = all moves playable)
        return 1.0 - (extra_balanced / extra_possible)

    white_sharpness = calculate_score(white_balanced, white_min, white_max)
    black_sharpness = calculate_score(black_balanced, black_min, black_max)

    # Calculate combined sharpness using harmonic mean
    if white_sharpness is None or black_sharpness is None:
        combined_sharpness = None
    else:
        combined_sharpness = harmonic_mean(white_sharpness, black_sharpness)

    return Sharpness(
        white=white_sharpness,
        black=black_sharpness,
        total=combined_sharpness,
    )


def convert_tree_to_nested_set(
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
        child_nodes, child_right = convert_tree_to_nested_set(
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
    analysis_tree: PositionNode,
) -> list[TreeNode]:
    """Convert the analysis tree to a list of TreeNode objects"""
    dfrc_id = chess960_to_dfrc_uid(data.white_id, data.black_id)
    tree_nodes, _ = convert_tree_to_nested_set(analysis_tree, dfrc_id, data.cfg_id)
    return tree_nodes


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
        dfrc_id=chess960_to_dfrc_uid(data.params.white_id, data.params.black_id),
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
        mirrored=is_mirrored(white, black),
        flipped=is_flipped(white, black),
    )


if __name__ == "__main__":
    # Example usage:
    from pathlib import Path

    from src.db import ParquetDatabase

    # Create a database for analysis results
    analysis_db = ParquetDatabase[AnalysisResult](
        model_class=AnalysisResult,
        parquet_file=Path("data/analysis_results.parquet"),
    )

    # Create a database for tree nodes
    tree_db = ParquetDatabase[TreeNode](
        model_class=TreeNode,
        parquet_file=Path("data/tree_nodes.parquet"),
    )

    # Load sample data from JSON file
    with open("analysis/11000.json", "rb") as f:
        sample_data: AnalysisData = msgspec.json.decode(f.read(), type=AnalysisData)

    # Convert tree to table
    tree_nodes = convert_analysis_tree(sample_data.params, sample_data.analysis_tree)

    sharpness = calculate_sharpness_score(
        tree_nodes,
        load_config(sample_data.params.cfg_id),
    )

    # Convert to AnalysisResult
    analysis_result = build_analysis_result(
        sample_data,
        sharpness,
    )

    # Store in database
    analysis_db.append([analysis_result])
    tree_db.append(tree_nodes)
