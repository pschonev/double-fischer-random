from dataclasses import dataclass
import datetime
import json
import logging
from typing import Any, Self

import pyarrow.json as pj
import pyarrow.parquet as pq

from scripts.utils import (
    get_prs,
    is_collaborator,
    add_label,
    remove_label,
    run_command,
    close_pr,
    PRAnalysisLabel,
    RESULTS_NDJSON,
    RESULTS_PARQUET,
    POSITIONS_FILE,
    WIP_FILE,
)

STALE_HOURS = 6  # hours until PR is considered stale


@dataclass
class AnalysisResult:
    branch_name: str
    pr_number: str
    position: str
    result: dict[str, Any]

    @classmethod
    def from_result(
        cls,
        pr_branch_name: str,
        pr_number: str,
        result: dict[str, Any],
    ) -> Self:
        return cls(
            branch_name=pr_branch_name,
            pr_number=pr_number,
            position=result["position"],
            result=result,
        )


def process_prs() -> None:
    try:
        prs = get_prs()
        analysis_results: list[AnalysisResult] = []
        for pr in prs:
            try:
                pr_number = pr["number"]
                pr_user = pr["user"]["login"]
                pr_labels = [label["name"] for label in pr["labels"]]
                pr_branch_name = pr["head"]["ref"]
                pr_created_at = datetime.datetime.strptime(
                    pr["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).astimezone(datetime.UTC)

                # Remove stale PRs
                if datetime.datetime.now(
                    tz=datetime.UTC
                ) - pr_created_at > datetime.timedelta(hours=STALE_HOURS):
                    close_pr(pr_number)
                    continue

                # Process "analyzed" PRs
                if PRAnalysisLabel.ANALYZED in pr_labels:
                    if is_collaborator(pr_user):
                        analysis_results.append(
                            AnalysisResult.from_result(
                                pr_branch_name=pr_branch_name,
                                pr_number=pr_number,
                                result=collect_analysis_results(pr_branch_name),
                            )
                        )
                    else:
                        add_label(pr_number, PRAnalysisLabel.READY_FOR_VALIDATION)
                        remove_label(pr_number, PRAnalysisLabel.ANALYZED)

                # Process "validated" PRs
                if PRAnalysisLabel.VALIDATED in pr_labels:
                    analysis_results.append(
                        AnalysisResult.from_result(
                            pr_branch_name=pr_branch_name,
                            pr_number=pr_number,
                            result=collect_analysis_results(pr_branch_name),
                        )
                    )

            except Exception:
                logging.exception(f"Error processing PR #{pr_number}")
                add_label(pr_number, PRAnalysisLabel.ERROR)

        try:
            push_analysis_results(analysis_results)
            convert_ndjson_to_parquet()
            for result in analysis_results:
                close_pr(result.pr_number)
        except Exception as e:
            logging.exception("Error pushing analysis results")
            raise e
    except Exception as e:
        logging.exception("Error fetching PRs")
        raise e


def collect_analysis_results(branch_name: str) -> dict[str, Any]:
    """Collect the analysis results for a given branch

    Args:
        branch_name: The branch name

    Returns:
        The analysis results

    Raises:
        Exception: If an error occurs while collecting the analysis results
    """
    try:
        # Checkout the branch and fetch the latest changes
        run_command(["git", "fetch", "origin", branch_name])
        run_command(["git", "checkout", branch_name])

        with WIP_FILE.open("r") as file:
            return json.load(file)

    except Exception as e:
        logging.exception(f"Error collecting analysis results for branch {branch_name}")
        raise e


def push_analysis_results(analysis_results: list[AnalysisResult]) -> None:
    """Push the analysis results to the NDJSON file and delete all analyzed positions
    from the positions.txt file.

    Args:
        analysis_results: The analysis results to push
    """
    run_command(["git", "checkout", "main"])
    run_command(["git", "pull", "origin", "main"])

    # Append new results to the NDJSON file
    analyzed_positions: set[str] = set()
    with RESULTS_NDJSON.open("a") as file:
        for result in analysis_results:
            file.write(json.dumps(result.result) + "\n")
            analyzed_positions.add(result.position)

    # Remove all analyzed positions from the positions file
    with POSITIONS_FILE.open("r") as file:
        positions = file.readlines()

    with POSITIONS_FILE.open("w") as file:
        for position in positions:
            if position.strip() not in analyzed_positions:
                file.write(position)

    # Commit the NDJSON file and the positions file
    run_command(["git", "add", str(RESULTS_NDJSON), str(POSITIONS_FILE)])
    run_command(
        [
            "git",
            "commit",
            "-m",
            f"chore(data): Updated {RESULTS_NDJSON} and {POSITIONS_FILE}",
        ]
    )
    run_command(["git", "push", "origin", "main"])


def convert_ndjson_to_parquet() -> None:
    try:
        # Read the NDJSON file into a Table
        table = pj.read_json(RESULTS_NDJSON)

        # Convert to Parquet
        pq.write_table(table, RESULTS_PARQUET, compression="BROTLI")

        # Commit the Parquet file
        run_command(["git", "add", str(RESULTS_PARQUET)])
        run_command(["git", "commit", "-m", f"chore(data): Updated {RESULTS_PARQUET}"])
        run_command(["git", "push", "origin", "main"])

    except Exception as e:
        logging.exception("Error converting NDJSON to Parquet")
        raise e


if __name__ == "__main__":
    process_prs()
