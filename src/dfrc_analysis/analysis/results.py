from dataclasses import dataclass

import msgspec


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
