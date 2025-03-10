# Example usage:
from pathlib import Path

import msgspec

from dfrc_analysis.analysis.config import load_config
from dfrc_analysis.analysis.eval import calculate_sharpness_score
from dfrc_analysis.analysis.results import AnalysisData
from dfrc_analysis.db.build_models import build_analysis_result, convert_analysis_tree
from dfrc_analysis.db.models import AnalysisResult, TreeNode
from dfrc_analysis.db.parquet_db import ParquetDatabase

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
