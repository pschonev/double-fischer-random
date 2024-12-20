from typing import Self

import chess.engine
import msgspec

from enum import IntEnum
import functools
from typing import Dict

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
    centipawn: int
    children: list[Self]


class AnalysisData(msgspec.Struct):
    """Contains the minimum data for an analysis"""

    dfrc_id: int

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


class BlunderPotential(msgspec.Struct):
    """Contains the blunder potential of a position
    The total is a combination of the white and black blunder potential"""

    white: float
    black: float
    total: float


class AnalysisResult(AnalysisData):
    """Contains the result of an analysis with the scores.

    Attributes:
        playability_score: The harmonic mean of the sharpness_score and the win_loss_ratio
    """
    white_id: int
    black_id: int
    
    white: str
    black: str

    cfg: AnalysisConfig

    playability_score: float

    balance_score: float
    sharpness: Sharpness

    wdl: chess.engine.Wdl

    blunder_potential: BlunderPotential

    PV: list[str]
    
    symmetric: bool
    mirrored: bool

    @classmethod
    def from_analysis_data(cls, data: AnalysisData) -> Self:
        """Create an AnalysisResult from an AnalysisData instance"""

        # load the config from the configs
        cfg = load_config(data.cfg_id)

        wdl = chess.engine.Cp(data.analysis_tree.centipawn).wdl()
        balance_score = cls._calculate_balance_score(wdl)
        sharpness = cls._calculate_sharpness_score(
            data.analysis_tree,
            cfg.balanced_threshold,
        )

        return cls(
            white=data.white,
            black=data.black,
            analyzer=data.analyzer,
            validator=data.validator,
            cfg_id=data.cfg_id,
            cfg=cfg,
            analysis_tree=data.analysis_tree,
            balance_score=balance_score,
            sharpness=sharpness,
            wdl=wdl,
            blunder_potential=cls._calculate_blunder_potential(data.analysis_tree),
            symmetric=cls._is_symmetric(data.white, data.black),
            mirrored=cls._is_mirrored(data.white, data.black),
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
                1
                for child in node.children
                if abs(child.centipawn) <= balanced_threshold
            )

            # Update accumulator for the appropriate player
            acc = black_acc if white else white_acc
            acc.balanced_moves += balanced_moves
            acc.total_positions += len(node.children)

            # Recurse for each child
            for child in node.children:
                if abs(child.centipawn) <= balanced_threshold:
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

    @staticmethod
    def _calculate_blunder_potential(
        analysis_tree: PositionNode,
        blunder_threshold: float = 2.0,  # Consider moves losing 2+ pawns as blunders
    ) -> BlunderPotential:
        """Calculate the blunder potential of a position

        Blunder potential considers:
        1. How many moves are blunders (eval drops by threshold)
        2. How severe the blunders are (magnitude of eval drops)
        3. The ratio of blunders to total legal moves

        Args:
            analysis_tree: The analysis tree containing position evaluations
            blunder_threshold: Minimum eval drop (in pawns) to consider a move a blunder

        Returns:
            BlunderPotential containing white, black and combined blunder potentials
        """

        class BlunderAccumulator(msgspec.Struct):
            total_blunder_severity: float = 0.0  # Sum of all eval drops
            blunder_count: int = 0  # Number of moves that are blunders
            total_moves: int = 0  # Total number of moves analyzed

        white_acc = BlunderAccumulator()
        black_acc = BlunderAccumulator()

        def calculate_position_blunder_potential(
            node: PositionNode,
            white: bool,
            ply: int = 0,
        ) -> None:
            if not node.children:
                return

            # Get best evaluation as baseline
            best_eval = (
                max(child.centipawn for child in node.children) / 100.0
            )  # Convert to pawns

            acc = white_acc if white else black_acc
            acc.total_moves += len(node.children)

            # Check each move for blunders
            for child in node.children:
                move_eval = child.centipawn / 100.0  # Convert to pawns
                eval_drop = best_eval - move_eval

                if eval_drop >= blunder_threshold:
                    acc.blunder_count += 1
                    acc.total_blunder_severity += eval_drop

            # Recurse for each child that isn't a clear blunder
            for child in node.children:
                move_eval = child.centipawn / 100.0
                if best_eval - move_eval < blunder_threshold:
                    calculate_position_blunder_potential(
                        child,
                        not white,  # Switch sides
                        ply + 1,
                    )

        # Start recursion from root
        calculate_position_blunder_potential(analysis_tree, white=True)

        def calculate_final_potential(acc: BlunderAccumulator) -> float:
            if acc.total_moves == 0:
                return 0.0

            # Combine three factors:
            # 1. Average severity of blunders
            severity = (
                acc.total_blunder_severity / acc.blunder_count
                if acc.blunder_count > 0
                else 0
            )

            # 2. Ratio of blunders to total moves
            blunder_ratio = acc.blunder_count / acc.total_moves

            # 3. Raw number of blunder opportunities (scaled)
            blunder_opportunities = min(acc.blunder_count / 5, 1.0)  # Cap at 5 blunders

            # Combine factors with weights
            return (
                0.4 * severity  # How bad are the blunders
                + 0.4 * blunder_ratio  # What fraction of moves are blunders
                + 0.2 * blunder_opportunities  # Raw blunder count (capped)
            )

        white_potential = calculate_final_potential(white_acc)
        black_potential = calculate_final_potential(black_acc)

        # Calculate combined potential using harmonic mean
        total_potential = (
            harmonic_mean(white_potential, black_potential)
            if white_potential + black_potential > 0
            else 0.0
        )

        return BlunderPotential(
            white=white_potential,
            black=black_potential,
            total=total_potential,
        )
