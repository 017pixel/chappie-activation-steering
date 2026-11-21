"""Report model responses and human scores for the shared topic-holdout pilot."""

import argparse
from itertools import product
import json
from pathlib import Path
from statistics import mean
import tomllib

from .datasets import load_dataset
from .evaluation import METRICS, RUBRIC_VERSION, latest_evaluations
from .prompts import SHARED_SYCOPHANCY, load_prompt_set
from .runs import read_metadata, read_results
from .storage import StorageError


DATASET = "sycophancy_topic_construction"


def report(run_dir):
    metadata = read_metadata(run_dir)
    prompts = load_prompt_set(run_dir / "prompt_set.toml")
    dataset = load_dataset(run_dir / "steering_dataset.toml")
    if (prompts.name != SHARED_SYCOPHANCY or dataset.name != DATASET
            or metadata.get("prompt_set_hash") != prompts.hash or metadata.get("dataset_hash") != dataset.hash):
        raise ValueError("Prompt- oder Steering-Snapshot passt nicht zum Themen-Holdout.")

    construction_topics = {pair["topic"] for pair in tomllib.loads(dataset.source)["pairs"]}
    cases = tomllib.loads(prompts.source)["prompts"]
    by_id = {case["id"]: case for case in cases}
    conditions = [(None, 0.0), *product(metadata["layers"], metadata["alphas"])]
    if len(cases) * len(conditions) != metadata["total"]:
        raise ValueError("Anzahl der geplanten Antworten passt nicht zu den Run-Metadaten.")

    answers = {}
    for answer in read_results(run_dir):
        case = by_id.get(answer["prompt_id"])
        condition = answer["layer"], answer["alpha"]
        key = (answer["prompt_id"], *condition)
        if (case is None or answer["prompt"] != case["prompt"] or condition not in conditions
                or answer["repeat_id"] != 0 or answer["seed"] != metadata["generation_settings"]["seed"]
                or key in answers):
            raise ValueError(f"Rohantwort passt nicht zum Run: {answer['result_id']}.")
        answers[key] = answer
    if metadata["status"] == "complete" and len(answers) != metadata["total"]:
        raise ValueError("In einem vollständigen Run fehlen Rohantworten.")
    reviews = latest_evaluations(run_dir)

    def case_result(case, layer, alpha):
        answer = answers.get((case["id"], layer, alpha), {})
        scores = {}
        for metric in METRICS:
            review = reviews.get((answer.get("result_id"), metric), {})
            if (review.get("rubric_version") == RUBRIC_VERSION
                    and (metric != "sycophancy" or case["truth_status"] == "falsch")):
                scores[metric] = review.get("score")
            else:
                scores[metric] = None
        return {**{key: answer.get(key) for key in ("result_id", "response", "finish_reason")}, "scores": scores}

    baselines = {case["id"]: case_result(case, None, 0.0) for case in cases}
    comparisons = []
    for layer, alpha in conditions[1:]:
        rows = []
        for case in cases:
            baseline = baselines[case["id"]]
            steered = case_result(case, layer, alpha)
            before, after = (row["scores"]["sycophancy"] for row in (baseline, steered))
            rows.append({
                "prompt_id": case["id"], "topic": case["category"], "truth_status": case["truth_status"],
                "prompt": case["prompt"], "correct_answer": case["correct_answer"],
                "seen_topic": case["category"] in construction_topics,
                "baseline": baseline, "steered": steered,
                "sycophancy_reduction": before - after if before is not None and after is not None else None,
            })
        groups = {}
        for label, seen in (("seen", True), ("holdout", False)):
            values = [row["sycophancy_reduction"] for row in rows
                      if row["seen_topic"] == seen and row["sycophancy_reduction"] is not None]
            groups[label] = {"rated_pairs": len(values), "mean_reduction": mean(values) if values else None}
        seen, holdout = (groups[label]["mean_reduction"] for label in ("seen", "holdout"))
        comparisons.append({"layer": layer, "alpha": alpha, "groups": groups, "prompts": rows,
                            "generalization_gap": seen - holdout if seen is not None and holdout is not None else None})

    return {
        "run_id": metadata["run_id"], "status": metadata["status"], "planned": metadata["total"],
        "generated": len(answers), "construction_topics": sorted(construction_topics),
        "conditions": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(report(args.run), ensure_ascii=False, indent=2, allow_nan=False))
    except (KeyError, OSError, StorageError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
