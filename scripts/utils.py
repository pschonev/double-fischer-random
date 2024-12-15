from enum import StrEnum
import os
from http import HTTPStatus
import logging
from pathlib import Path
import subprocess

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "pschonev"
REPO_NAME = "double-fischer-random"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

RESULTS_NDJSON = Path("results.ndjson")
RESULTS_PARQUET = Path("results.parquet")
WIP_FILE = Path(".wip_analysis.json")
POSITIONS_FILE = Path("positions.txt")
BRANCH_NAME_PATTERN = "analysis/{position}-{depth}"


class PRAnalysisLabel(StrEnum):
    ERROR = "error"
    ANALYSIS_IN_PROGRESS = "analysis-in-progress"
    ANALYZED = "analyzed"  # this distincton is made so that collaborators' PRs are merged automatically
    READY_FOR_VALIDATION = "ready-for-validation"
    VALIDATION_IN_PROGRESS = "validation-in-progress"
    VALIDATED = "validated"
    MANUAL_VALIDATION_REQUIRED = "manual-validation-required"


def get_prs() -> dict:
    response = requests.get(f"{API_URL}/pulls", headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json()


def close_pr(pr_number: str) -> None:
    try:
        requests.patch(
            f"{API_URL}/pulls/{pr_number}",
            headers=HEADERS,
            json={"state": "closed"},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logging.exception(f"Error closing PR #{pr_number}")
        raise e


def is_collaborator(username: str) -> bool:
    response = requests.get(
        f"{API_URL}/collaborators/{username}", headers=HEADERS, timeout=10
    )
    return response.status_code == HTTPStatus.NO_CONTENT


def add_label(pr_number: str, label: PRAnalysisLabel) -> None:
    try:
        requests.post(
            f"{API_URL}/issues/{pr_number}/labels",
            headers=HEADERS,
            json={"labels": [label]},
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logging.exception(f"Error adding label '{label}' to PR #{pr_number}")
        raise e


def remove_label(pr_number: str, label: PRAnalysisLabel) -> None:
    try:
        requests.delete(
            f"{API_URL}/issues/{pr_number}/labels/{label}",
            headers=HEADERS,
            timeout=10,
        ).raise_for_status()
    except Exception as e:
        logging.exception(f"Error removing label '{label}' from PR #{pr_number}")
        raise e


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logging.exception(f"Command '{command}' failed with error: {e.stderr}")
        raise e
    else:
        return result.stdout
