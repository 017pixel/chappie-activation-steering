"""Export blind review rows or pair human ratings from one Issue #4 run."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import tomllib

from activation_steering.evaluation import METRICS, RUBRIC_VERSION, latest_evaluations
from activation_steering.runs import read_metadata, read_results
from activation_steering.storage import StorageError, read_jsonl


STANCES = ("agree", "disagree", "unclear", "qualified")
COMPARISONS = {
    "taylor_swift": ("taylor_swift", "anonymous"),
    "marie_curie": ("marie_curie", "anonymous"),
    "barack_obama": ("barack_obama", "anonymous"),
    "fictional": ("name", "anonymous"),
    "fictional_title": ("title", "fictional"),
}
CSV_FIELDS = (
    "comparison", "claim_id", "claim_category", "claim_truth", "claim_difficulty",
    "person_category", "name_pair_id", "identity_category_left", "identity_category_right",
    "authority_role_left", "authority_role_right", "speaker_mode_left", "speaker_mode_right",
    "prompt_variant_left", "prompt_variant_right", "layer", "alpha", "seed",
    "prompt_id_left", "prompt_id_right", "result_id_left", "result_id_right",
    "stance_left", "stance_right", "agreement_delta", "agreement_delta_minus_baseline",
    "false_disagreement_delta", "false_disagreement_delta_minus_baseline",
) + tuple(f"{metric}_{side}" for metric in METRICS for side in ("left", "right")) + (
    "truth_status", "speaker_id_left", "speaker_id_right",
    "qualification_delta", "qualification_delta_minus_baseline",
)


def read_run(run_dir):
    metadata = read_metadata(run_dir)
    source = (run_dir / "prompt_set.toml").read_bytes()
    if sha256(source).hexdigest() != metadata["prompt_set_hash"]:
        raise ValueError("Der Prompt-Snapshot stimmt nicht mit dem gespeicherten Hash überein.")
    snapshot = tomllib.loads(source.decode("utf-8"))
    if snapshot.get("identity_study_schema") not in (1, 2):
        raise ValueError("Der Run benötigt einen Identitätsstudien-Prompt-Snapshot mit Schema 1 oder 2.")
    prompts = {item["id"]: item for item in snapshot["prompts"]}
    conditions = [(None, 0.0), *product(metadata["layers"], metadata["alphas"])]
    condition_set = set(conditions)
    if len(prompts) != len(snapshot["prompts"]) or len(prompts) * len(conditions) != metadata["total"]:
        raise ValueError("Prompt-IDs oder Anzahl der Bedingungen passen nicht zum Run.")

    results = {}
    seed = metadata["generation_settings"]["seed"]
    for row in read_results(run_dir):
        prompt_id = row["prompt_id"]
        condition = (row["layer"], row["alpha"])
        key = (prompt_id, condition)
        if (prompt_id not in prompts or row["prompt"] != prompts[prompt_id]["prompt"]
                or condition not in condition_set or row["repeat_id"] != 0 or row["seed"] != seed
                or key in results):
            raise ValueError(f"Rohantwort passt nicht zum eingefrorenen Run: {row['result_id']}.")
        results[key] = row
    if metadata["status"] == "complete" and len(results) != metadata["total"]:
        raise ValueError("In einem vollständigen Run fehlen Rohantworten.")
    return metadata, prompts, conditions, results


def review_id(metadata, result_id):
    return sha256(f"{metadata['run_id']}:{metadata['prompt_set_hash']}:{result_id}".encode()).hexdigest()


def export_review(output, metadata, prompts, results):
    rows = []
    for result in results.values():
        prompt = prompts[result["prompt_id"]]
        rows.append({
            "review_id": review_id(metadata, result["result_id"]),
            "claim": prompt["claim_text"],
            "correct_answer": prompt["correct_answer"],
            "response": result["response"],
            "stance": None,
            "reason": "",
        })
    with output.open("x", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["review_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_stances(path, metadata, results):
    if not path.is_file():
        raise ValueError(f"Bewertungsdatei fehlt: {path}.")
    ids = {review_id(metadata, result["result_id"]): result["result_id"] for result in results.values()}
    stances = {}
    for row in read_jsonl(path):
        key, stance, reason = row.get("review_id"), row.get("stance"), row.get("reason")
        if not isinstance(key, str) or key not in ids or ids[key] in stances:
            raise ValueError(f"Unbekannte oder doppelte Review-ID: {key}.")
        if ("stance" not in row or (stance is not None and
                (stance not in STANCES or not isinstance(reason, str) or not reason.strip()))):
            raise ValueError(f"Ungültige Bewertung für {key}.")
        stances[ids[key]] = stance
    return stances


def matched_pairs(prompts):
    if all("truth_status" in prompt for prompt in prompts.values()):
        by_slot = {(p["claim_id"], p["speaker_id"]): p for p in prompts.values()}
        for (claim, speaker), left in by_slot.items():
            if speaker in COMPARISONS:
                name, other = COMPARISONS[speaker]
                yield name, left, by_slot[claim, other]
        return

    # Earlier frozen runs use the original identity and wording combinations.
    by_slot = {
        (p["claim_id"], p["speaker_id"], p["authority_role"], p["speaker_mode"], p["prompt_variant"]): p
        for p in prompts.values()
    }
    for (claim, speaker, role, mode, variant), left in by_slot.items():
        if left["identity_category"] == "prominent":
            unknown = f"{left['name_pair_id']}-unknown"
            yield "prominence", left, by_slot[claim, unknown, role, mode, variant]
        if role == "professor_math":
            yield "title", left, by_slot[claim, speaker, "guest", mode, variant]
        if left["identity_category"] == "unknown":
            yield "name", left, by_slot[claim, "anonymous", role, mode, variant]
        if mode == "self":
            yield "speaker_mode", left, by_slot[claim, speaker, role, "third", variant]
        if variant == "identity_first":
            yield "word_order", left, by_slot[claim, speaker, role, mode, "claim_first"]


def stance_delta(left, right, event):
    if left is None or right is None:
        return None
    return int(left == event) - int(right == event)


def paired_rows(prompts, conditions, results, seed, stances, evaluations):
    for comparison, left, right in matched_pairs(prompts):
        truth_status = left.get("truth_status", "richtig" if left.get("claim_truth") else "falsch")
        events = {"qualification": "qualified"} if truth_status == "offen" else {"agreement": "agree"}
        if truth_status == "richtig":
            events["false_disagreement"] = "disagree"
        baselines = {}
        for condition in conditions:
            layer, alpha = condition
            result_ids = [
                results.get((prompt["id"], condition), {}).get("result_id")
                for prompt in (left, right)
            ]
            left_stance, right_stance = (stances.get(result_id) for result_id in result_ids)
            row = {
                "comparison": comparison, "claim_id": left["claim_id"],
                "claim_category": left["category"], "truth_status": truth_status,
                "claim_truth": (truth_status == "richtig") if truth_status != "offen" else None,
                "claim_difficulty": left.get("claim_difficulty"), "person_category": left.get("person_category"),
                "name_pair_id": left.get("name_pair_id"), "layer": layer, "alpha": alpha, "seed": seed,
            }
            for metric, event in events.items():
                delta = stance_delta(left_stance, right_stance, event)
                row[f"{metric}_delta"] = delta
                if layer is None:
                    baselines[metric] = delta
                elif delta is not None and baselines[metric] is not None:
                    row[f"{metric}_delta_minus_baseline"] = delta - baselines[metric]
            for side, prompt, result_id, stance in zip(("left", "right"), (left, right), result_ids,
                                                        (left_stance, right_stance), strict=True):
                for label in ("identity_category", "authority_role", "speaker_mode", "prompt_variant", "speaker_id"):
                    row[f"{label}_{side}"] = prompt.get(label)
                row[f"prompt_id_{side}"] = prompt["id"]
                row[f"result_id_{side}"] = result_id
                row[f"stance_{side}"] = stance
                for metric in METRICS:
                    review = evaluations.get((result_id, metric), {})
                    if ((metric != "sycophancy" or truth_status == "falsch")
                            and review.get("rubric_version") == RUBRIC_VERSION):
                        row[f"{metric}_{side}"] = review.get("score")
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("export", "pair"):
        subparser = commands.add_parser(command)
        subparser.add_argument("run", type=Path)
        subparser.add_argument("output", type=Path)
        if command == "pair":
            subparser.add_argument("--reviews", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata, prompts, conditions, results = read_run(args.run)
        if args.command == "export":
            export_review(args.output, metadata, prompts, results)
        else:
            stances = load_stances(args.reviews, metadata, results)
            evaluations = latest_evaluations(args.run)
            rows = paired_rows(prompts, conditions, results, metadata["generation_settings"]["seed"],
                               stances, evaluations)
            with args.output.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
    except (KeyError, OSError, StorageError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
