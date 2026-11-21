from collections import Counter, defaultdict
import csv
from itertools import product
from math import isqrt
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

from activation_steering.datasets import load_dataset
from activation_steering.evaluation import latest_evaluations
from activation_steering.prompts import load_prompt_set
from activation_steering.runs import plan_run
from activation_steering.storage import append_jsonl, atomic_json, read_jsonl
from scripts.identity_prompts import MODES, STUDY_PATH, VARIANTS, build_prompts, write_prompt_set
from scripts.identity_review import export_review, load_stances, matched_pairs, paired_rows, read_run


def normalize(text):
    return "".join(re.findall(r"\w+", text.lower()))


class IdentityStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = tomllib.loads(STUDY_PATH.read_text(encoding="utf-8"))
        cls.full = build_prompts(cls.study, "full")
        cls.pilot = build_prompts(cls.study, "pilot")

    def test_balance_and_original_claims(self):
        truth_and_difficulty = list(product((True, False), ("easy", "subtle")))
        topics = ("mathematics", "physics", "general_knowledge")
        self.assertEqual(Counter((c["category"], c["truth"], c["difficulty"]) for c in self.study["claims"]),
                         Counter({(topic, truth, difficulty): 2 for topic in topics
                                  for truth, difficulty in truth_and_difficulty}))
        self.assertEqual(Counter((c["truth"], c["difficulty"]) for c in self.study["claims"]
                                 if c["id"] in self.study["pilot_claims"]), Counter(dict.fromkeys(truth_and_difficulty, 2)))
        self.assertEqual(int("101", 2), 5)
        self.assertGreater(3 * 2, 1 * 4)
        self.assertEqual(sum(range(1, 40, 2)), 400)
        self.assertEqual(sum(all(n % d for d in range(2, isqrt(n) + 1)) for n in range(2, 51)), 15)

        training = load_dataset("sycophancy")
        texts = [normalize(pair[side]) for pair in training.pairs for side in ("target", "contrast")]
        for claim in self.study["claims"]:
            self.assertTrue(claim["basis"], claim["id"])
            if claim["category"] != "mathematics":
                self.assertTrue(claim["source_url"].startswith("https://"))
            self.assertFalse(any(normalize(claim["text"]) in text for text in texts), claim["id"])

    def test_complete_controls_and_pilot_subset(self):
        for pair in self.study["name_pairs"]:
            self.assertEqual(len(pair["prominent"].split()), 2)
            self.assertEqual(len(pair["unknown"].split()), 2)
            self.assertLessEqual(abs(len(pair["prominent"]) - len(pair["unknown"])), 1)
        full_ids = {p["id"] for p in self.full}
        pilot_ids = {p["id"] for p in self.pilot}
        self.assertEqual(len(self.full), len(full_ids))
        self.assertEqual(len(self.pilot), len(pilot_ids))
        self.assertEqual(self.pilot, [p for p in self.full if p["id"] in pilot_ids])
        for prompts, claim_count, counts in (
            (self.pilot, 8, {"prominent": 4, "unknown": 4, "anonymous": 1}),
            (self.full, 24, {"prominent": 8, "unknown": 8, "anonymous": 1}),
        ):
            per_claim = defaultdict(list)
            for item in prompts:
                per_claim[item["claim_id"]].append(item)
            self.assertEqual(len(per_claim), claim_count)
            for items in per_claim.values():
                slots = defaultdict(set)
                claims = {p["claim_text"] for p in items}
                self.assertEqual(len(claims), 1)
                self.assertEqual(len({p["correct_answer"] for p in items}), 1)
                for item in items:
                    slots[item["speaker_id"], item["authority_role"]].add((item["speaker_mode"], item["prompt_variant"]))
                    self.assertTrue(item["prompt"].startswith("Fiktives Szenario. "))
                    self.assertIn(f"„{item['claim_text']}.“", item["prompt"])
                    if item["identity_name"]:
                        self.assertIn(item["identity_name"], item["prompt"])
                self.assertEqual(Counter(item["identity_category"] for item in items if item["authority_role"] == "none"),
                                 Counter({group: count * 4 for group, count in counts.items()}))
                self.assertTrue(all(cells == set(product(MODES, VARIANTS)) for cells in slots.values()))
            comparisons = Counter(name for name, _, _ in matched_pairs({p["id"]: p for p in prompts}))
            self.assertEqual(comparisons["prominence"], claim_count * counts["prominent"] * 4)
            self.assertEqual(comparisons["title"], claim_count * (2 if prompts is self.pilot else 9) * 4)
            self.assertEqual(comparisons["name"], claim_count * (24 if prompts is self.pilot else 96))
            self.assertEqual(comparisons["speaker_mode"], len(prompts) // 2)
            self.assertEqual(comparisons["word_order"], len(prompts) // 2)
            people = Counter(p["identity_name"] for p in prompts
                             if p["identity_name"] and p["authority_role"] == "none")
            self.assertEqual(len(people), 16)
            self.assertEqual(set(people.values()), {16 if prompts is self.pilot else 96})

    def test_pilot_rotates_the_titled_name_through_all_eight_pairs(self):
        titles = Counter(p["name_pair_id"] for p in self.pilot
                         if p["identity_category"] == "unknown" and p["authority_role"] == "professor_math")
        self.assertEqual(titles, Counter({pair["id"]: 4 for pair in self.study["name_pairs"]}))

    def test_prompt_loader_and_run_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            for scope, count in (("pilot", 416), ("full", 3360)):
                path, built = write_prompt_set(scope, Path(directory))
                loaded = load_prompt_set(path.stem, path.parent)
                self.assertEqual(loaded, load_prompt_set(write_prompt_set(scope, Path(directory))[0]))
                self.assertEqual((built, len(loaded.prompts)), (count, count))
                self.assertIn("claim_truth", loaded.source)
                self.assertIn("identity_study_schema = 1", loaded.source)
                self.assertNotIn("claim_truth", loaded.prompts[0])
                config = {"layers": [8, 32, 56], "alphas": [-1, 1, 2],
                          "prompts": [p["prompt"] for p in loaded.prompts],
                          "prompt_ids": [p["id"] for p in loaded.prompts]}
                plan = plan_run(config, 64)
                self.assertEqual((plan["baseline"], plan["total"]), (count, count * 10))
                self.assertEqual(plan_run({**config, "alphas": [0]}, 64)["total"], count)
                self.assertEqual(plan_run({**config, "layers": "all"}, 64)["total"], count * 193)

    def test_blind_labels_and_pairing_with_missing_responses_and_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, count = write_prompt_set("pilot", root)
            path.rename(root / "prompt_set.toml")
            metadata = {"run_id": "fixture-run", "status": "interrupted", "total": count * 3,
                        "artifact_hashes": {}, "backend_fingerprint": {}, "layers": [8], "alphas": [-1, 1],
                        "prompt_set_hash": load_prompt_set(root / "prompt_set.toml").hash,
                        "generation_settings": {"seed": 42}}
            atomic_json(root / "metadata.json", metadata)
            source = {p["id"]: p for p in self.pilot}
            _, left, right = next(pair for pair in matched_pairs(source)
                                  if pair[0] == "prominence" and not pair[1]["claim_truth"])
            _, true_left, true_right = next(pair for pair in matched_pairs(source)
                                            if pair[0] == "prominence" and pair[1]["claim_truth"])
            for prompt, layer, alpha, stance in (
                (left, None, 0, "agree"), (right, None, 0, "unclear"),
                (left, 8, -1, "disagree"), (right, 8, -1, "agree"), (left, 8, 1, None),
                (true_left, None, 0, "disagree"), (true_right, None, 0, "unclear"),
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
            self.assertEqual(len(exported), 7)
            self.assertEqual(set(exported[0]), {"review_id", "claim", "correct_answer", "response", "stance", "reason"})
            for row in exported:
                if row["response"] != "fixture":
                    append_jsonl(root / "rated.jsonl", {**row, "stance": row["response"], "reason": "Fixture"})
            stances = load_stances(root / "rated.jsonl", actual_metadata, results)
            rows = [row for row in paired_rows(prompts, conditions, results, 42, stances, latest_evaluations(root))
                    if row["comparison"] == "prominence" and row["prompt_id_left"] == left["id"]]
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
                true_baseline = next(row for row in csv.DictReader(handle) if row["comparison"] == "prominence"
                                     and row["prompt_id_left"] == true_left["id"] and row["layer"] == "")
            self.assertEqual(true_baseline["agreement_delta"], "0")
            self.assertEqual(true_baseline["false_disagreement_delta"], "1")
