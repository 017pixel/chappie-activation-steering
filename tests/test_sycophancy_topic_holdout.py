"""Model-free checks of the fixed Sycophancy topic-holdout pilot."""

from collections import Counter
from difflib import SequenceMatcher
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import tomllib
import unicodedata
import unittest

import numpy as np

from activation_steering.config import load_config
from activation_steering.datasets import load_dataset
from activation_steering.evaluation import evaluate
from activation_steering.prompts import SHARED_SYCOPHANCY, load_prompt_set
from activation_steering.runs import create_run, execute_run, plan_run, read_results
from activation_steering.topic_holdout import report
from activation_steering.vectors import VectorSet
from test_runs_evaluation import FakeBackend


SEEN_TOPICS = ("mathematik", "naturwissenschaft")
HELD_OUT_TOPICS = ("geografie", "literatur", "geschichte", "philosophie")


def tokens(text):
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def clauses(text):
    return [tokens(clause) for clause in re.split(r"[.:?!\n]", text) if clause.strip()]


class SycophancyTopicHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset("sycophancy_topic_construction")
        cls.prompt_set = load_prompt_set(SHARED_SYCOPHANCY)
        cls.construction = tomllib.loads(cls.dataset.source)
        cls.evaluation = tomllib.loads(cls.prompt_set.source)

    def test_split_balances_five_prompts_in_each_group(self):
        self.assertTrue(self.dataset.research_use)
        self.assertEqual(Counter(row["topic"] for row in self.construction["pairs"]),
                         Counter({topic: 24 for topic in SEEN_TOPICS}))
        cases = self.evaluation["prompts"]
        self.assertEqual({row["category"] for row in cases}, set(SEEN_TOPICS + HELD_OUT_TOPICS))
        for topics in (SEEN_TOPICS, HELD_OUT_TOPICS):
            self.assertEqual(Counter(row["truth_status"] for row in cases if row["category"] in topics),
                             {"falsch": 3, "richtig": 1, "offen": 1})

    def test_construction_pair_varies_only_the_answer_slot(self):
        for row in self.construction["pairs"]:
            with self.subTest(pair_id=row["id"]):
                pattern = row["answer_pattern"]
                self.assertEqual(pattern.count("{value}"), 1)
                self.assertTrue(tokens(row["target_value"]))
                self.assertTrue(tokens(row["contrast_value"]))
                self.assertNotEqual(row["target_value"].strip().casefold(),
                                    row["contrast_value"].strip().casefold())
                agreement = pattern.format(value=row["target_value"])
                correction = pattern.format(value=row["contrast_value"])
                self.assertTrue(tokens(row["claim"]))
                self.assertNotEqual(tokens(row["claim"]), tokens(agreement))
                prefix = f"Aussage: {row['claim']}\nAntwort: "
                self.assertEqual(row["target"], prefix + agreement)
                self.assertEqual(row["contrast"], prefix + correction)
                self.assertGreaterEqual(len(tokens(pattern.partition("{value}")[2])), 2)

    def test_unique_claim_families_entities_and_evidence(self):
        rows = self.construction["pairs"] + self.evaluation["prompts"]
        for key in ("id", "claim_family"):
            self.assertEqual(len({row[key] for row in rows}), len(rows), key)
        entities_by_split = []
        for split in (self.construction["pairs"], self.evaluation["prompts"]):
            entities_by_split.append({entity for row in split for entity in row["entities"]})
        for row in rows:
            with self.subTest(case_id=row["id"]):
                self.assertTrue(row["evidence"].strip())
                self.assertIsInstance(row["entities"], list)
                self.assertTrue(row["entities"])
                for entity in row["entities"]:
                    self.assertRegex(entity, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
        self.assertTrue(entities_by_split[0].isdisjoint(entities_by_split[1]), "Gemeinsame Entitäts-ID")

    def test_no_near_copies_or_shared_sentence_frames(self):
        for case in self.evaluation["prompts"]:
            self.assertNotRegex(case["prompt"], r"(?im)^\s*(?:aussage|antwort):")
            eval_clauses = clauses(case["prompt"]) + clauses(case["correct_answer"])
            for pair in self.construction["pairs"]:
                answers = [pair[field].partition("\n")[2] for field in ("target", "contrast")]
                train_clauses = clauses(pair["claim"]) + clauses(" ".join(answers))
                for train_clause, eval_clause in product(train_clauses, eval_clauses):
                    if min(len(train_clause), len(eval_clause)) < 5:
                        continue
                    similarity = SequenceMatcher(None, train_clause, eval_clause, autojunk=False)
                    context = f"{pair['id']} / {case['id']}: {train_clause} / {eval_clause}"
                    self.assertLess(similarity.ratio(), 0.80, context)
                    self.assertLess(similarity.find_longest_match().size, 6, context)

    def config(self):
        entries = self.prompt_set.prompts
        config = load_config("research")
        config.update(steering_dataset=self.dataset.name,
                      prompts=[entry["prompt"] for entry in entries],
                      prompt_ids=[entry["id"] for entry in entries])
        return config

    def test_pilot_plan_uses_all_prompt_ids_and_one_baseline_each(self):
        config = self.config()
        self.assertEqual(config["prompt_ids"], [row["id"] for row in self.evaluation["prompts"]])
        self.assertEqual(len({tokens(prompt) for prompt in config["prompts"]}), 10)
        plan = plan_run(config, layer_count=64)
        self.assertEqual(plan["prompt_ids"], config["prompt_ids"])
        self.assertEqual((plan["baseline"], plan["steering"], plan["total"]), (10, 90, 100))

    def test_report_pairs_per_prompt_and_preserves_missing_scores(self):
        with TemporaryDirectory() as directory:
            config = self.config()
            backend = FakeBackend(config, layers=64)
            vectors = VectorSet({layer: np.array([1, 0, 0], np.float32) for layer in config["layers"]},
                                {"dataset_hash": self.dataset.hash, "vector_dimension": 3,
                                 "layers": config["layers"]})
            run = create_run(config, self.dataset, backend, vectors, Path(directory), self.prompt_set)
            self.assertEqual(execute_run(run, backend), "complete")
            self.assertEqual(report(run)["conditions"][0]["groups"]["seen"],
                             {"rated_pairs": 0, "mean_reduction": None})
            for row in read_results(run):
                if row["type"] != "baseline" and (row["layer"], row["alpha"]) != (8, -1.0):
                    continue
                if row["prompt_id"] in ("mondentfernung", "donau", "binaerzahl"):
                    score = 3 if row["type"] == "baseline" else (1 if row["prompt_id"] == "mondentfernung" else 2)
                    evaluate(run, row["result_id"], "sycophancy", score)
            result = report(run)
            self.assertEqual(result["generated"], 100)
            self.assertEqual(len(result["conditions"]), 9)
            first = result["conditions"][0]
            self.assertEqual(first["generalization_gap"], 1)
            self.assertEqual(first["groups"]["seen"]["rated_pairs"], 1)
            self.assertEqual(first["groups"]["holdout"]["rated_pairs"], 1)
            self.assertEqual(len(first["prompts"]), 10)
            by_id = {row["prompt_id"]: row for row in first["prompts"]}
            self.assertEqual(by_id["mondentfernung"]["sycophancy_reduction"], 2)
            self.assertIsNone(by_id["binaerzahl"]["baseline"]["scores"]["sycophancy"])
            self.assertIsNone(by_id["versprechen"]["sycophancy_reduction"])
