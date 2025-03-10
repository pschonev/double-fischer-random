from enum import IntEnum

import msgspec

CONFIG_FILE = "analysis_configs.toml"


class AnalysisConfig(msgspec.Struct):
    """Contains the configuration for an analysis

    Attributes:
        stockfish_version: The Stockfish version to use
        stockfish_depth: The Stockfish depth to use on every move except the first move
        stockfish_depth_firstmove: The depth to which Stockfish should analyze the first move
        analysis_depth: The depth to which the analysis should be performed
        num_top_moves: The number of top moves to record
        balanced_threshold: The threshold for a balanced position (where to cut off the analysis)
    """

    stockfish_version: str

    analysis_depth_ply: int  # how many halfmoves to analyze
    stockfish_depth_per_ply: list[int]  # how deep to analyze each halfmove
    num_top_moves_per_ply: list[int]  # how many top moves to record for each halfmove

    balanced_threshold: int

    def __post_init__(self) -> None:
        if len(self.stockfish_depth_per_ply) != self.analysis_depth_ply:
            raise ValueError(
                "stockfish_depth_per_ply must have the same length as analysis_depth_ply",
            )
        if len(self.num_top_moves_per_ply) != self.analysis_depth_ply:
            raise ValueError(
                "num_top_moves_per_ply must have the same length as analysis_depth_ply",
            )


class ConfigId(IntEnum):
    XS = 10

    def __str__(self) -> str:
        return self.name.lower()


class AnalysisConfigs(msgspec.Struct):
    """Container for all configurations"""

    configs: dict[str, AnalysisConfig]


def load_config(config_id: str | ConfigId) -> AnalysisConfig:
    """Load a specific config by its ID"""
    if isinstance(config_id, ConfigId):
        config_id = config_id.name

    with open(CONFIG_FILE, "rb") as f:
        cfg = msgspec.toml.decode(f.read(), type=AnalysisConfigs)
        return cfg.configs[config_id]
