import msgspec

from enum import IntEnum
import functools
from typing import Dict

# Configuration file path
ANALYSIS_CONFIGS_PATH = "analysis_configs.toml"


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

    stockfish_depth: int
    stockfish_depth_firstmove: int

    analysis_depth: int
    num_top_moves: int

    balanced_threshold: int


class ConfigId(IntEnum):
    XS = 10
    S = 20
    M = 30
    L = 40
    XL = 50

    def __str__(self) -> str:
        return self.name.lower()


class AnalysisConfigs(msgspec.Struct):
    """Container for all analysis configurations"""

    configs: Dict[ConfigId, AnalysisConfig]


@functools.cache
def load_configs(path: str = ANALYSIS_CONFIGS_PATH) -> AnalysisConfigs:
    """Load all configs from TOML file"""
    with open(path, "rb") as f:
        return msgspec.toml.decode(f.read())


def load_config(config_id: str | ConfigId) -> AnalysisConfig:
    """Load a specific config by its ID"""
    if isinstance(config_id, str):
        config_id = ConfigId(config_id)
    return load_configs().configs[config_id]
