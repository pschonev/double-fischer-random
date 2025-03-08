import argparse
import logging
import os
from pathlib import Path

import msgspec

from src.analysis_config import ConfigId
from src.analysis_results import AnalysisData, AnalysisParams
from src.positions import dfrc_to_chess960_uids
from src.stockfish_analyzer import analyse_dfrc_position

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a DFRC position")
    parser.add_argument(
        "dfrc_id",
        type=int,
        help="Starting DFRC ID to analyze (between 0 and 960*960-1)",
    )
    parser.add_argument(
        "-n",
        "--num_positions",
        type=int,
        default=1,
        help="Number of sequential positions to analyze",
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=[c.name for c in ConfigId],
        default="XS",
        help="Analysis configuration to use",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=4096,
        help="Hash size in MB for Stockfish",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use for Stockfish",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    available_cpus = os.process_cpu_count() or 1
    threads: int = args.threads or max(1, available_cpus - 2)
    hash_size: int = args.hash or 1024

    # Validate DFRC ID
    max_id = 960 * 960 - 1
    if not 0 <= args.dfrc_id <= max_id:
        parser.error(f"DFRC ID must be between 0 and {max_id}")

    # Validate that the range doesn't exceed max_id
    if args.dfrc_id + args.num_positions - 1 > max_id:
        parser.error(f"The range of positions exceeds the maximum ID of {max_id}")

    analysis_folder = Path("analysis")
    analysis_folder.mkdir(exist_ok=True)

    for i in range(args.num_positions):
        current_dfrc_id = args.dfrc_id + i
        logging.info(
            f"Analyzing position {current_dfrc_id} ({i + 1}/{args.num_positions})",
        )

        white_id, black_id = dfrc_to_chess960_uids(current_dfrc_id)
        params = AnalysisParams(
            white_id=white_id,
            black_id=black_id,
            cfg_id=args.config,
            hash=hash_size,
            threads=threads,
        )

        # Run analysis
        analysis_tree = analyse_dfrc_position(
            params=params,
            verbose=args.verbose,
        )

        # Convert to JSON and save
        output_file = analysis_folder / f"{current_dfrc_id}.json"

        analysis_data = AnalysisData(
            params=params,
            analyzer="test_analyzer",
            analysis_tree=analysis_tree,
        )
        json_data = msgspec.json.encode(analysis_data)

        with open(output_file, "wb") as f:
            f.write(json_data)

        logging.info(f"Analysis saved to {output_file}")


if __name__ == "__main__":
    main()
