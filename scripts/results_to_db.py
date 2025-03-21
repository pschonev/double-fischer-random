import argparse
from pathlib import Path

import msgspec

from dfrc_analysis.analysis.config import load_config
from dfrc_analysis.analysis.eval import calculate_position_sharpness
from dfrc_analysis.analysis.results import AnalysisData
from dfrc_analysis.db.build_models import build_analysis_result, convert_analysis_tree
from dfrc_analysis.db.models import AnalysisResult, TreeNode
from dfrc_analysis.db.parquet_db import ParquetDatabase

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--eval-threshold",
    "-t",
    type=int,
    help="Threshold for evaluating sharpness",
)
args = argparser.parse_args()

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

# Path to the directory containing JSON files
json_directory = Path(__file__).resolve().parent.parent / "analysis"

# Process each JSON file in the directory
for json_file in json_directory.glob("*.json"):
    # Load data from JSON file
    with json_file.open("rb") as f:
        sample_data: AnalysisData = msgspec.json.decode(f.read(), type=AnalysisData)

    # Convert tree to table
    tree_nodes = convert_analysis_tree(sample_data.params, sample_data.analysis_tree)

    # Calculate sharpness score
    sharpness = calculate_position_sharpness(
        tree_nodes,
        load_config(sample_data.params.cfg_id),
        args.eval_threshold,
    )

    # Convert to AnalysisResult
    analysis_result = build_analysis_result(
        sample_data,
        sharpness,
        args.eval_threshold,
    )

    # Store in databases
    analysis_db.append([analysis_result])
    tree_db.append(tree_nodes)
