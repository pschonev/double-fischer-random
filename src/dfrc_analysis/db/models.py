from datetime import datetime

from sqlalchemy import ARRAY, Column, String
from sqlmodel import Field, PrimaryKeyConstraint, SQLModel


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
    balance_score: float
    playability_score: float | None = None

    # Position properties
    mirrored: bool
    flipped: bool
    swapped_id: int

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
