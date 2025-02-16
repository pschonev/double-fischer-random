from typing import Self

import chess.engine
import msgspec

from enum import IntEnum
import functools
from typing import Dict

from src.positions import get_chess960_position, chess960_uid

# Configuration file path
ANALYSIS_CONFIGS_PATH = "analysis_configs.toml"


class AnalysisConfig(msgspec.Struct):
    """Contains the configuration for an analysis

    Attributes:
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


def harmonic_mean(a: float, b: float) -> float:
    """Calculate the harmonic mean of two numbers"""
    return 2 * (a * b) / (a + b)


class PositionNode(msgspec.Struct):
    """Node in the position analysis tree which represents a halfmove (ply) and its evaluation"""

    move: str
    cpl: int
    children: list[Self]
    pv: list[str]


class AnalysisData(msgspec.Struct):
    """Contains the minimum data for an analysis"""

    white_id: int
    black_id: int

    analyzer: str
    validator: str

    cfg_id: str

    analysis_tree: PositionNode


class Sharpness(msgspec.Struct):
    """Contains the sharpness of a position
    The total is a combination of the white and black sharpness"""

    white: float
    black: float
    total: float


class AnalysisResult(AnalysisData):
    """Contains the result of an analysis with the scores.

    Attributes:
        playability_score: The harmonic mean of the sharpness_score and the win_loss_ratio
    """

    dfrc_id: int

    white_id: int
    black_id: int

    white: str
    black: str

    cfg: AnalysisConfig

    playability_score: float

    balance_score: float
    sharpness: Sharpness

    wdl: chess.engine.Wdl

    symmetric: bool
    mirrored: bool

    @classmethod
    def from_analysis_data(cls, data: AnalysisData) -> Self:
        """Create an AnalysisResult from an AnalysisData instance"""

        # load the config from the configs
        cfg = load_config(data.cfg_id)

        wdl = chess.engine.Cp(data.analysis_tree.cpl).wdl()

        white = get_chess960_position(data.white_id)
        black = get_chess960_position(data.black_id)

        balance_score = cls._calculate_balance_score(wdl)
        sharpness = cls._calculate_sharpness_score(
            data.analysis_tree,
            cfg.balanced_threshold,
        )

        return cls(
            white_id=data.white_id,
            black_id=data.black_id,
            white=white,
            black=black,
            dfrc_id=chess960_uid(data.white_id, data.black_id),
            pv=data.analysis_tree.pv,
            analyzer=data.analyzer,
            validator=data.validator,
            cfg_id=data.cfg_id,
            cfg=cfg,
            analysis_tree=data.analysis_tree,
            balance_score=balance_score,
            sharpness=sharpness,
            wdl=wdl,
            symmetric=cls._is_symmetric(white, black),
            mirrored=cls._is_mirrored(white, black),
            playability_score=harmonic_mean(
                balance_score,
                sharpness.total,
            ),
        )

    @staticmethod
    def _is_symmetric(white: str, black: str) -> bool:
        """Check if a position is symmetric"""
        return white == black

    @staticmethod
    def _is_mirrored(white: str, black: str) -> bool:
        """Check if a position is mirrored"""
        return white == black[::-1]

    @staticmethod
    def _calculate_balance_score(wdl: chess.engine.Wdl) -> float:
        """Calculate the balance score from the win, draw, loss probabilities

        Args:
            wdl: The win, draw, loss probabilities

        Returns:
            A float between 0 and 1 representing the balance of the position.
            0 is perfectly balanced, 1 is completely unbalanced.
        """
        return abs(wdl.wins - wdl.losses) / 1000

    @staticmethod
    def _calculate_sharpness_score(
        analysis_tree: PositionNode,
        balanced_threshold: int,
    ) -> Sharpness:
        """Calculate the sharpness score of a position

        Args:
            analysis_tree: The analysis tree
            balanced_threshold: The threshold for a balanced position

        Returns:
            Sharpness tuple containing white, black and combined sharpness scores
        """

        class SharpnessAccumulator(msgspec.Struct):
            balanced_moves: int = 0
            total_positions: int = 0

        white_acc = SharpnessAccumulator()
        black_acc = SharpnessAccumulator()

        def _calculate_sharpness(
            node: PositionNode,
            white: bool,
            ply: int = 0,
        ) -> None:
            """Recursively calculate the sharpness score of a position"""

            # Skip empty nodes
            if not node.children:
                return

            # Count balanced moves at this position
            balanced_moves = sum(
                1 for child in node.children if abs(child.cpl) <= balanced_threshold
            )

            # Update accumulator for the appropriate player
            acc = black_acc if white else white_acc
            acc.balanced_moves += balanced_moves
            acc.total_positions += len(node.children)

            # Recurse for each child
            for child in node.children:
                if abs(child.cpl) <= balanced_threshold:
                    _calculate_sharpness(
                        child,
                        not white,  # Switch sides
                        ply + 1,
                    )

        # Start recursion from root
        _calculate_sharpness(analysis_tree, white=True)

        # Calculate final sharpness scores
        def calculate_final_score(acc: SharpnessAccumulator) -> float:
            if acc.total_positions == 0:
                return 0.0
            # Return inverted ratio (1 - balanced_ratio) to get sharpness
            return 1.0 - (acc.balanced_moves / acc.total_positions)

        white_sharpness = calculate_final_score(white_acc)
        black_sharpness = calculate_final_score(black_acc)

        # Calculate combined sharpness using harmonic mean
        if white_sharpness + black_sharpness == 0:
            combined_sharpness = 0.0
        else:
            combined_sharpness = harmonic_mean(white_sharpness, black_sharpness)

        return Sharpness(white_sharpness, black_sharpness, combined_sharpness)
