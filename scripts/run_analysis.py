import json
from src.analysis import analyze_position


def main():
    with open(".wip_analysis.json") as f:
        data = json.load(f)

    position = data["position"]
    analysis_result = analyze_position(position)

    data["analysis"] = analysis_result

    with open(".wip_analysis.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
