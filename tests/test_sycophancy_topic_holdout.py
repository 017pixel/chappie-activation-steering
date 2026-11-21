"""Model-free checks of the fixed Sycophancy topic-holdout pilot."""

from collections import Counter
from difflib import SequenceMatcher
from itertools import product
import re
import tomllib
import unicodedata
import unittest

from activation_steering.datasets import load_dataset
from activation_steering.prompts import load_prompt_set
from activation_steering.runs import plan_run


SEEN_TOPICS = ("mathematik", "naturwissenschaft")
HELD_OUT_TOPICS = ("geschichte", "digitale_technik")
TRUTH_CLASSES = ("falsch", "richtig", "mehrdeutig")
WORDING = ("direkte_behauptung", "hoefliche_frage",
           "selbstsichere_behauptung", "bestaetigungsbitte")


def tokens(text):
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def clauses(text):
    return [tokens(clause) for clause in re.split(r"[.:?!\n]", text) if clause.strip()]


class SycophancyTopicHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset("sycophancy_topic_construction")
        cls.prompt_set = load_prompt_set("sycophancy_topic_holdout")
        cls.construction = tomllib.loads(cls.dataset.source)
        cls.evaluation = tomllib.loads(cls.prompt_set.source)

    def test_split_and_balanced_formulations(self):
        self.assertEqual(self.construction["study_revision"], 2)
        self.assertEqual(self.evaluation["study_revision"], 2)
        self.assertEqual(tuple(self.evaluation["seen_topics"]), SEEN_TOPICS)
        self.assertEqual(tuple(self.evaluation["held_out_topics"]), HELD_OUT_TOPICS)
        self.assertTrue(self.dataset.research_use)
        self.assertEqual(Counter(row["topic"] for row in self.construction["pairs"]),
                         Counter({topic: 24 for topic in SEEN_TOPICS}))
        observed = Counter((row["category"], row["truth_status"], row["pressure"])
                           for row in self.evaluation["prompts"])
        self.assertEqual(observed, Counter({cell: 2 for cell in product(
            SEEN_TOPICS + HELD_OUT_TOPICS, TRUTH_CLASSES, WORDING)}))

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
        entities = []
        for row in rows:
            with self.subTest(case_id=row["id"]):
                self.assertTrue(row["evidence"].strip())
                self.assertIsInstance(row["entities"], list)
                self.assertTrue(row["entities"])
                for entity in row["entities"]:
                    self.assertRegex(entity, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
                    entities.append(entity)
        self.assertEqual(len(set(entities)), len(entities), "Gemeinsame Entitäts-ID")

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

    def test_pilot_plan_uses_all_prompt_ids_and_one_baseline_each(self):
        entries = self.prompt_set.prompts
        config = {
            "layers": [8, 32, 56], "alphas": [-1.0, 1.0, 2.0],
            "prompts": [entry["prompt"] for entry in entries],
            "prompt_ids": [entry["id"] for entry in entries],
        }
        self.assertEqual(config["prompt_ids"], [row["id"] for row in self.evaluation["prompts"]])
        self.assertEqual(len({tokens(prompt) for prompt in config["prompts"]}), 96)
        plan = plan_run(config, layer_count=64)
        self.assertEqual(plan["prompt_ids"], config["prompt_ids"])
        self.assertEqual((plan["baseline"], plan["steering"], plan["total"]), (96, 864, 960))
