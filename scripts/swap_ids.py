import json
from pathlib import Path


def swap_ids_in_json_files(directory: Path) -> None:
    # Create a Path object for the given directory
    directory_path = Path(directory)

    # Iterate over all JSON files in the directory
    for filepath in directory_path.glob("*.json"):
        # Open and read the JSON file
        with filepath.open("r") as file:
            data = json.load(file)

        # Swap white_id and black_id
        params = data.get("params", {})
        white_id = params.get("white_id")
        black_id = params.get("black_id")

        if white_id is not None and black_id is not None:
            params["white_id"], params["black_id"] = black_id, white_id

        # Save the updated JSON back to the same file
        with filepath.open("w") as file:
            json.dump(data, file, indent=4)


# Calculate the path to the /analysis directory relative to the scripts directory
directory = Path(__file__).resolve().parent.parent / "analysis"

# Call the function to swap IDs in JSON files
swap_ids_in_json_files(directory)
