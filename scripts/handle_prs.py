import datetime
import json
import logging
import os
import subprocess
from http import HTTPStatus
from pathlib import Path

import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "pschonev"
REPO_NAME = "double-fischer-random"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}


def get_prs() -> dict:
    response = requests.get(f"{API_URL}/pulls", headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json()


def is_collaborator(username: str) -> bool:
    response = requests.get(
        f"{API_URL}/collaborators/{username}", headers=HEADERS, timeout=10
    )
    return response.status_code == HTTPStatus.NO_CONTENT


def process_prs() -> None:
    try:
        prs = get_prs()
        for pr in prs:
            try:
                pr_number = pr["number"]
                pr_user = pr["user"]["login"]
                pr_labels = [label["name"] for label in pr["labels"]]
                pr_created_at = datetime.datetime.strptime(
                    pr["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).astimezone(datetime.UTC)

                # Remove stale PRs
                if datetime.datetime.now(
                    tz=datetime.UTC
                ) - pr_created_at > datetime.timedelta(hours=6):
                    close_pr(pr_number)
                    continue

                # Process "analyzed" PRs
                if "analyzed" in pr_labels:
                    if is_collaborator(pr_user):
                        merge_results(pr_number)
                    else:
                        add_label(pr_number, "ready_to_validate")
                        remove_label(pr_number, "analyzed")

                # Process "validated" PRs
                if "validated" in pr_labels:
                    merge_results(pr_number)

            except Exception:
                logging.exception(f"Error processing PR #{pr_number}")
                add_label(pr_number, "error")
    except Exception:
        logging.exception("Error fetching PRs")


def close_pr(pr_number: str) -> None:
    try:
        requests.patch(
            f"{API_URL}/pulls/{pr_number}",
            headers=HEADERS,
            json={"state": "closed"},
            timeout=10,
        ).raise_for_status()
    except Exception:
        logging.exception(f"Error closing PR #{pr_number}")


def add_label(pr_number: str, label: str) -> None:
    try:
        requests.post(
            f"{API_URL}/issues/{pr_number}/labels",
            headers=HEADERS,
            json={"labels": [label]},
            timeout=10,
        ).raise_for_status()
    except Exception:
        logging.exception(f"Error adding label '{label}' to PR #{pr_number}")


def remove_label(pr_number: str, label: str) -> None:
    try:
        requests.delete(
            f"{API_URL}/issues/{pr_number}/labels/{label}",
            headers=HEADERS,
            timeout=10,
        ).raise_for_status()
    except Exception:
        logging.exception(f"Error removing label '{label}' from PR #{pr_number}")


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        logging.exception(f"Command '{command}' failed with error: {e.stderr}")
        raise
    else:
        return result.stdout


def merge_results(pr_number: str) -> None:
    try:
        # Fetch the PR details to get the branch name
        pr_details = requests.get(
            f"{API_URL}/pulls/{pr_number}",
            headers=HEADERS,
            timeout=10,
        ).json()
        branch_name = pr_details["head"]["ref"]

        # Checkout the branch and fetch the .wip.json file
        run_command(["git", "fetch", "origin", branch_name])
        run_command(["git", "checkout", branch_name])

        # Use pathlib for file paths
        wip_file_path = Path(".wip.json")
        ndjson_file_path = Path("result.ndjson")

        if wip_file_path.exists():
            with wip_file_path.open("r") as file:
                wip_data = json.load(file)

            # Append the data to result.ndjson
            with ndjson_file_path.open("a") as ndjson_file:
                json.dump(wip_data, ndjson_file)
                ndjson_file.write("\n")

            # Remove the .wip.json file
            wip_file_path.unlink()

            # Commit the changes
            run_command(["git", "add", "result.ndjson"])
            run_command(
                ["git", "commit", "-m", f"Merged analysis data from PR #{pr_number}"]
            )
            run_command(["git", "push", "origin", "main"])

            # Merge the PR
            requests.put(
                f"{API_URL}/pulls/{pr_number}/merge",
                headers=HEADERS,
                json={"commit_title": f"Merge PR #{pr_number}"},
                timeout=10,
            ).raise_for_status()

        # Convert result.ndjson to Parquet
        convert_ndjson_to_parquet(ndjson_file_path)

    except Exception:
        logging.exception(f"Error merging results for PR #{pr_number}")
        add_label(pr_number, "error")


def convert_ndjson_to_parquet(ndjson_file_path: Path) -> None:
    try:
        # Read the NDJSON file into a DataFrame
        df_results = pd.read_json(ndjson_file_path, lines=True)

        # Convert to Parquet
        parquet_file_path = Path("result.parquet")
        df_results.to_parquet(parquet_file_path)

        # Commit the Parquet file
        run_command(["git", "add", str(parquet_file_path)])
        run_command(["git", "commit", "-m", "Updated result.parquet"])
        run_command(["git", "push", "origin", "main"])

    except Exception:
        logging.exception("Error converting NDJSON to Parquet")


if __name__ == "__main__":
    process_prs()
