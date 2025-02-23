import json
import os
import sys
from typing import Optional, Tuple
import logging
from textwrap import dedent

import polars as pl
from github import Github
from cyclopts import App, Parameter

from src.analysis_config import ConfigId

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MIN_POSITIONS = 10


def get_github_client():
    """Initialize GitHub client with token"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error(
            dedent("""
            GitHub token not found!
            Please set GITHUB_TOKEN environment variable:
            export GITHUB_TOKEN=your_token_here
        """)
        )
        sys.exit(1)

    return Github(token)


def get_config_level(config_id: str) -> int:
    """Convert config_id string to numeric level"""
    try:
        return int(ConfigId[config_id.upper()]) if config_id else 0
    except (KeyError, AttributeError):
        return 0


def get_available_positions(repo, config_id: ConfigId) -> pl.DataFrame:
    """Load positions and check their analysis status"""
    # Use streaming for large file
    positions_df = pl.scan_csv("positions.csv")

    # Get positions in open PRs
    reserved = []
    pulls = repo.get_pulls(state="open")
    for pr in pulls:
        if pr.head.ref.startswith("analysis/"):
            try:
                start, end = map(int, pr.head.ref.split("/")[1].split("_"))
                reserved.extend(range(start, end + 1))
            except ValueError:
                continue

    reserved_series = pl.Series("reserved", reserved)

    # Build query
    query = (
        positions_df.with_columns(
            [
                pl.col("config_id")
                .map_elements(get_config_level)
                .alias("config_level"),
                # Position is available if config_id is null or existing level is lower than requested
                (
                    (pl.col("config_id").is_null())
                    | (
                        pl.col("config_id").map_elements(get_config_level)
                        < int(config_id)
                    )
                ).alias("config_available"),
            ]
        )
        # Mark positions as unavailable if they're in reserved list
        .with_columns(pl.col("config_available") & ~pl.col("id").is_in(reserved_series))
        .alias("available")
    )

    # Execute query and collect results
    return query.collect()


def find_next_available_range(
    positions_df: pl.DataFrame,
    positions: int,
    start_id: Optional[int],
    config_id: ConfigId,
) -> Tuple[int, int]:
    """Find the next available range of positions"""
    if positions < MIN_POSITIONS:
        logger.error(
            dedent(f"""
            Error: Minimum number of positions required is {MIN_POSITIONS}.
            You requested: {positions}

            Please request at least {MIN_POSITIONS} positions to create a PR.
            This requirement ensures efficient use of reviewer time and
            maintains consistent analysis batch sizes.
        """)
        )
        sys.exit(1)

    # Get available positions as sorted list
    available_positions = (
        positions_df.filter(pl.col("available"))
        .select("id")
        .sort("id")
        .get_column("id")
        .to_list()
    )

    if not available_positions:
        logger.error(
            dedent(f"""
            No available positions found for config_id: {config_id}

            This might be because:
            1. All positions are already analyzed at this level or higher
            2. All remaining positions are reserved in open PRs
        """)
        )
        sys.exit(1)

    if start_id is not None:
        if start_id not in available_positions:
            current_config = (
                positions_df.filter(pl.col("id") == start_id).select("config_id").item()
            )

            if current_config:
                logger.error(
                    dedent(f"""
                    Starting position {start_id} already analyzed with config_id: {current_config}
                    Requested config_id: {config_id}

                    Cannot analyze with same or lower config_id.
                """)
                )
            else:
                logger.error(f"Starting position {start_id} is not available!")
            sys.exit(1)

        # Find consecutive positions from start_id
        start_idx = available_positions.index(start_id)
        if start_idx + positions > len(available_positions):
            logger.error(
                dedent(f"""
                Not enough available positions from {start_id}!
                Available positions: {len(available_positions[start_idx:])}
                Requested positions: {positions}
            """)
            )
            sys.exit(1)

        consecutive = available_positions[start_idx : start_idx + positions]
        if len(consecutive) != positions or consecutive != list(
            range(consecutive[0], consecutive[-1] + 1)
        ):
            logger.error(
                dedent(f"""
                No consecutive range of {positions} positions available from {start_id}!
                This might be because some positions in the range are:
                - Already analyzed at this level or higher
                - Reserved in open PRs
            """)
            )
            sys.exit(1)

        return consecutive[0], consecutive[-1]
    else:
        # Find first available consecutive range using window functions
        consecutive_ranges = (
            pl.DataFrame({"id": available_positions})
            .with_columns(
                [
                    pl.col("id").diff().alias("diff"),
                    pl.col("id").shift(-positions + 1).alias("end_id"),
                ]
            )
            .filter(
                (pl.col("diff") == 1).fill_null(True) & pl.col("end_id").is_not_null()
            )
            .select(["id", "end_id"])
            .collect()
        )

        if consecutive_ranges.height == 0:
            logger.error(
                dedent(f"""
                No consecutive range of {positions} positions available!
                This might be because available positions are scattered
                and no consecutive range of {positions} positions exists.
            """)
            )
            sys.exit(1)

        # Take first range
        first_range = consecutive_ranges.row(0)
        return first_range[0], first_range[0] + positions - 1


def create_branch_and_pr(repo, start_id: int, end_id: int, config_id: ConfigId):
    """Create feature branch and PR with template files"""
    branch_name = f"analysis/{start_id}_{end_id}"

    try:
        # Create branch
        main_branch = repo.get_branch("main")
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=main_branch.commit.sha)

        # Create template files
        for id in range(start_id, end_id + 1):
            template = {
                "id": id,
                "config_id": str(config_id),
                "status": "pending",
                # Add other required fields
            }

            repo.create_file(
                path=f"contributions/{id}.json",
                message=f"Add template for ID {id}",
                content=json.dumps(template, indent=2),
                branch=branch_name,
            )

        # Create PR
        pr = repo.create_pull(
            title=f"Analysis: {start_id}-{end_id} (config: {config_id})",
            body=dedent(f"""
                Automated PR for analysis range {start_id}-{end_id}
                Configuration: {config_id}
                Number of positions: {end_id - start_id + 1}
            """),
            head=branch_name,
            base="main",
        )

        logger.info(
            dedent(f"""
            Successfully created PR #{pr.number}
            PR URL: {pr.html_url}
            Configuration: {config_id}

            Next steps:
            1. git fetch origin {branch_name}
            2. git checkout {branch_name}
            3. Add your analysis data to the template files
            4. Commit and push your changes
        """)
        )

    except Exception as e:
        logger.error(f"Failed to create branch/PR: {str(e)}")
        sys.exit(1)


app = App(name="setup_analysis")


@app.command()
def setup(
    positions: int = Parameter(default=10, help="Number of positions to analyze"),
    start_id: Optional[int] = Parameter(
        default=None, help="Starting position ID (optional)"
    ),
    config_id: ConfigId = Parameter(
        default=ConfigId.XS, help="Analysis configuration ID (XS, S, M, L, XL)"
    ),
):
    """Setup analysis branch and PR for a range of positions"""
    g = get_github_client()
    repo = g.get_repo("your-username/your-repo")  # Replace with your repo

    positions_df = get_available_positions(repo, config_id)
    start_id, end_id = find_next_available_range(
        positions_df, positions, start_id, config_id
    )
    create_branch_and_pr(repo, start_id, end_id, config_id)


if __name__ == "__main__":
    app()
