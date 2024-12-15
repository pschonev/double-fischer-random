import json
import subprocess


def get_github_user():
    result = subprocess.run(
        ["git", "config", "user.name"], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_prs_with_label(label):
    result = subprocess.run(
        ["gh", "pr", "list", "--label", label, "--json", "number,headRefName,author"],
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def checkout_branch(branch_name):
    subprocess.run(["git", "checkout", branch_name])


def label_pr(pr_number, label):
    subprocess.run(["gh", "pr", "edit", str(pr_number), "--add-label", label])


def create_branch(branch_name):
    subprocess.run(["git", "checkout", "-b", branch_name])


def create_pr(branch_name, title, body, labels):
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            branch_name,
            "--title",
            title,
            "--body",
            body,
            "--label",
            labels,
        ]
    )


def write_json_file(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    github_user = get_github_user()
    prs = get_prs_with_label("ready_for_validation")

    for pr in prs:
        if pr["author"]["login"] != github_user:
            checkout_branch(pr["headRefName"])
            label_pr(pr["number"], "validation_in_progress")
            data = {"validator": github_user}
            write_json_file(".wip_analysis.json", data)
            print(pr["headRefName"])
            return

    with open("positions.txt") as f:
        positions = f.readlines()

    for position in positions:
        position = position.strip()
        white, black, depth = position.split(",")
        branch_name = f"analysis/{white}_{black}-{depth}"
        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch_name], capture_output=True, text=True
        )
        if not result.stdout:
            create_branch(branch_name)
            data = {"analyzer": github_user, "position": position}
            write_json_file(".wip_analysis.json", data)
            create_pr(
                branch_name,
                f"Analysis for {position}",
                "Automated analysis PR",
                "analysis_in_progress",
            )
            print(branch_name)
            return


if __name__ == "__main__":
    main()
