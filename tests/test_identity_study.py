from collections import Counter
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

from activation_steering.config import load_config
from activation_steering.evaluation import latest_evaluations
from activation_steering.prompts import SHARED_SYCOPHANCY, load_prompt_set
from activation_steering.runs import plan_run
from activation_steering.storage import append_jsonl, atomic_json, read_jsonl
from scripts.identity_prompts import STUDY_PATH, build_prompts, write_prompt_set
from scripts.identity_review import export_review, load_stances, matched_pairs, paired_rows, read_run


PREFIXES = {
    "anonymous": "",
    "taylor_swift": "Fiktive Zuschreibung: Taylor Swift. ",
    "marie_curie": "Fiktive Zuschreibung: Marie Curie. ",
    "barack_obama": "Fiktive Zuschreibung: Barack Obama. ",
    "fictional": "Fiktive Zuschreibung: Maëlle Cerny. ",
    "fictional_title": "Fiktive Zuschreibung: Maëlle Cerny (Professorin für Physik). ",
}


class IdentityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = load_prompt_set(SHARED_SYCOPHANCY)
        study = tomllib.loads(STUDY_PATH.read_text(encoding="utf-8"))
        cls.pilot = build_prompts(study, cls.shared)

    def test_six_versions_only_change_the_prefix(self):
        by_case = {row["id"]: row for row in tomllib.loads(self.shared.source)["prompts"]}
        self.assertEqual(len({row["id"] for row in self.pilot}), 60)
        for row in self.pilot:
            shared = by_case[row["claim_id"]]
            self.assertEqual(row["prompt"], PREFIXES[row["speaker_id"]] + shared["prompt"])
            self.assertEqual(row["claim_text"], shared["prompt"])
            for field in ("correct_answer", "truth_status", "category", "claim_family"):
                self.assertEqual(row[field], shared[field])
        self.assertEqual(Counter((p["claim_id"], p["speaker_id"]) for p in self.pilot),
                         {(case_id, speaker): 1 for case_id in by_case for speaker in PREFIXES})

        pairs = list(matched_pairs({row["id"]: row for row in self.pilot}))
        self.assertEqual(Counter(name for name, _, _ in pairs),
                         {"taylor_swift": 10, "marie_curie": 10, "barack_obama": 10, "name": 10, "title": 10})
        for name, left, right in pairs:
            self.assertEqual(left["claim_id"], right["claim_id"])
            self.assertEqual(right["speaker_id"], "fictional" if name == "title" else "anonymous")

    def test_prompt_loader_and_run_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path, count = write_prompt_set(Path(directory))
            original = path.read_bytes()
            subprocess.run([sys.executable, "-m", "scripts.identity_prompts",
                            "--output-dir", directory], check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), original)
            loaded = load_prompt_set(path.stem, path.parent)
            self.assertEqual((count, len(loaded.prompts)), (60, 60))
            snapshot = tomllib.loads(loaded.source)
            self.assertEqual(snapshot["identity_study_schema"], 2)
            self.assertEqual(snapshot["base_prompt_set"], self.shared.name)
            self.assertEqual(snapshot["base_prompt_set_sha256"], self.shared.hash)
            config = load_config("research")
            config.update(prompt_set=loaded.name, prompts=[p["prompt"] for p in loaded.prompts],
                          prompt_ids=[p["id"] for p in loaded.prompts])
            plan = plan_run(config, 64)
            self.assertEqual((plan["baseline"], plan["steering"], plan["total"]), (60, 540, 600))

    def test_blind_labels_and_pairing_with_missing_responses_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, count = write_prompt_set(root)
            path.rename(root / "prompt_set.toml")
            metadata = {"run_id": "fixture-run", "status": "interrupted", "total": count * 3,
                        "artifact_hashes": {}, "backend_fingerprint": {}, "layers": [8], "alphas": [-1, 1],
                        "prompt_set_hash": load_prompt_set(root / "prompt_set.toml").hash,
                        "generation_settings": {"seed": 42}}
            atomic_json(root / "metadata.json", metadata)
            source = {p["id"]: p for p in self.pilot}
            _, left, right = next(pair for pair in matched_pairs(source)
                                  if pair[0] == "taylor_swift" and pair[1]["truth_status"] == "falsch")
            _, true_left, true_right = next(pair for pair in matched_pairs(source)
                                            if pair[0] == "taylor_swift" and pair[1]["truth_status"] == "richtig")
            _, open_left, open_right = next(pair for pair in matched_pairs(source)
                                            if pair[0] == "title" and pair[1]["truth_status"] == "offen")
            for prompt, layer, alpha, stance in (
                (left, None, 0, "agree"), (right, None, 0, "unclear"),
                (left, 8, -1, "disagree"), (right, 8, -1, "agree"), (left, 8, 1, None),
                (true_left, None, 0, "disagree"), (true_right, None, 0, "unclear"),
                (open_left, None, 0, "qualified"), (open_right, None, 0, "unclear"),
            ):
                result_id = f"{prompt['id']}-{alpha}"
                append_jsonl(root / "results.jsonl", {
                    "prompt_id": prompt["id"], "prompt": prompt["prompt"], "result_id": result_id,
                    "type": "baseline" if layer is None else "steering", "layer": layer,
                    "alpha": alpha, "seed": 42, "repeat_id": 0, "created_at": "fixture",
                    "response": stance or "fixture", "finish_reason": "eog", "prompt_tokens": 10,
                    "output_tokens": 1, "output_token_ids": [1], "ttft": 0.1, "generation_time": 0.1,
                    "total_time": 0.2, "tokens_per_second": 10,
                })
            for prompt, scores in ((left, [3, None]), (right, [0])):
                for score in scores:
                    append_jsonl(root / "evaluations.jsonl", {
                        "result_id": f"{prompt['id']}-0", "metric": "correctness", "score": score,
                        "reviewer_type": "human", "reviewed_at": "fixture", "rubric_version": "2",
                    })

            actual_metadata, prompts, conditions, results = read_run(root)
            export_review(root / "review.jsonl", actual_metadata, prompts, results)
            exported = read_jsonl(root / "review.jsonl")
            self.assertEqual(len(exported), 9)
            for row in exported:
                self.assertNotIn("Fiktive Zuschreibung", row["claim"])
            self.assertEqual(set(exported[0]), {"review_id", "claim", "correct_answer", "response", "stance", "reason"})
            for row in exported:
                if row["response"] != "fixture":
                    append_jsonl(root / "rated.jsonl", {**row, "stance": row["response"], "reason": "Fixture"})
            stances = load_stances(root / "rated.jsonl", actual_metadata, results)
            rows = [row for row in paired_rows(prompts, conditions, results, 42, stances, latest_evaluations(root))
                    if row["comparison"] == "taylor_swift" and row["prompt_id_left"] == left["id"]]
            baseline, contrast, missing = rows
            self.assertEqual(baseline["agreement_delta"], 1)
            self.assertIsNone(baseline["correctness_left"])
            self.assertEqual(baseline["correctness_right"], 0)
            self.assertEqual(contrast["agreement_delta"], -1)
            self.assertEqual(contrast["agreement_delta_minus_baseline"], -2)
            self.assertNotIn("false_disagreement_delta", baseline)
            self.assertIsNone(missing["stance_left"])
            self.assertIsNone(missing["result_id_right"])
            self.assertIsNone(missing["agreement_delta"])
            self.assertNotIn("agreement_delta_minus_baseline", missing)

            csv_path = root / "pairs.csv"
            subprocess.run([sys.executable, "-m", "scripts.identity_review", "pair", str(root), str(csv_path),
                            "--reviews", str(root / "rated.jsonl")], check=True, capture_output=True)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                true_baseline = next(row for row in csv.DictReader(handle) if row["comparison"] == "taylor_swift"
                                     and row["prompt_id_left"] == true_left["id"] and row["layer"] == "")
            self.assertEqual(true_baseline["agreement_delta"], "0")
            self.assertEqual(true_baseline["false_disagreement_delta"], "1")
