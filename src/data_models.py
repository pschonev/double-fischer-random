from typing import Optional
from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisTask(Base):
    """Tracks analysis assignments and their status"""

    __tablename__ = "analysis_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    dfrc_uid: Mapped[str] = mapped_column(String(16), index=True)
    analyzer: Mapped[str] = mapped_column(String(50))
    config_id: Mapped[str] = mapped_column(String(32))


class CurrentAnalysis(Base):
    """Temporary storage for positions being analyzed"""

    __tablename__ = "current_analysis"

    dfrc_uid: Mapped[str] = mapped_column(String(16), primary_key=True)
    config_id: Mapped[str] = mapped_column(String(32))


class ResultMain(Base):
    """Main results for each analyzed position"""

    __tablename__ = "result_main"

    dfrc_uid: Mapped[str] = mapped_column(String(16), primary_key=True)
    config_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    analyzer: Mapped[str] = mapped_column(String(50))
    threads: Mapped[int] = mapped_column(Integer)
    hash_size: Mapped[int] = mapped_column(Integer)
    is_flipped: Mapped[bool] = mapped_column(Boolean)
    is_mirrored: Mapped[bool] = mapped_column(Boolean)
    white_id: Mapped[int] = mapped_column(Integer)
    black_id: Mapped[int] = mapped_column(Integer)
    white: Mapped[str] = mapped_column(String(8))
    black: Mapped[str] = mapped_column(String(8))
    analysis_starting_pos: Mapped[str] = mapped_column(String(100))
    symmetry_score: Mapped[float] = mapped_column(Float)
    sharpness_score: Mapped[float] = mapped_column(Float)
    playability_score: Mapped[float] = mapped_column(Float)


class ResultNode(Base):
    """Analysis tree nodes using nested set pattern"""

    __tablename__ = "result_tree"

    # Primary key for individual node identification
    id: Mapped[int] = mapped_column(primary_key=True)

    # Tree identification (composite key)
    dfrc_uid: Mapped[str] = mapped_column(String(16))
    config_id: Mapped[str] = mapped_column(String(32))

    # Nested set pattern fields
    lft: Mapped[int] = mapped_column(Integer, index=True)
    rgt: Mapped[int] = mapped_column(Integer, index=True)

    # Node data
    move: Mapped[str] = mapped_column(String(10))
    centipawn_loss: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mate_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    principal_variation: Mapped[str] = mapped_column(String(500))

    __table_args__ = (
        # Ensure lft/rgt are unique within a tree
        UniqueConstraint("dfrc_uid", "config_id", "lft"),
        UniqueConstraint("dfrc_uid", "config_id", "rgt"),
    )
