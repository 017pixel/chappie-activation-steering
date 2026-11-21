"""Prompt-set validation without loading a model."""

from collections import Counter
from pathlib import Path
import tempfile
import tomllib
import unittest

from activation_steering.config import load_config
from activation_steering.datasets import load_dataset
from activation_steering.prompts import (
    SHARED_SYCOPHANCY, PromptSetError, discover_prompt_sets, load_prompt_set,
)


VALID = '''name = "fixture"
description = "Fixture prompts."
[[prompts]]
id = "p01"
category = "fakt"
pressure = "direkt"
prompt = "One?"
correct_answer = "One."
[[prompts]]
id = "p02"
category = "fakt"
pressure = "peer"
prompt = "Two?"
correct_answer = "Two."
'''


class PromptSetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, source):
        path = self.directory / name
        path.write_text(source, encoding="utf-8")
        return path

    def test_valid_prompt_set_and_hash(self):
        path = self.write("fixture.toml", VALID)
        prompt_set = load_prompt_set(path)
        self.assertEqual(prompt_set.name, "fixture")
        self.assertEqual([entry["id"] for entry in prompt_set.prompts], ["p01", "p02"])
        self.assertEqual(prompt_set.prompts[0]["pressure"], "direkt")
        self.assertEqual(len(prompt_set.hash), 64)

    def test_name_must_match_stem(self):
        path = self.write("other.toml", VALID)
        with self.assertRaisesRegex(PromptSetError, "Dateistem"):
            load_prompt_set("other", self.directory)

    def test_duplicate_and_missing_ids_are_rejected(self):
        duplicate = VALID.replace('id = "p02"', 'id = "p01"')
        path = self.write("duplicate.toml", duplicate)
        with self.assertRaisesRegex(PromptSetError, "doppelte ID"):
            load_prompt_set(path)
        missing = VALID.replace('correct_answer = "One."\n', "")
        path = self.write("missing.toml", missing)
        with self.assertRaisesRegex(PromptSetError, "correct_answer"):
            load_prompt_set(path)

    def test_discovery_skips_template_and_reports_invalid(self):
        self.write("fixture.toml", VALID)
        self.write("_template.toml", VALID.replace('name = "fixture"', 'name = "_template"'))
        self.write("broken.toml", "name =")
        found, errors = discover_prompt_sets(self.directory)
        self.assertEqual([item.name for item in found], ["fixture"])
        self.assertIn("broken.toml", errors)
        self.assertNotIn("_template.toml", errors)


class SharedPilotTests(unittest.TestCase):
    def test_shared_and_legacy_prompt_sets(self):
        config = load_config("research")
        self.assertEqual(config["prompt_set"], SHARED_SYCOPHANCY)
        shared = load_prompt_set(config["prompt_set"])
        entries = tomllib.loads(shared.source)["prompts"]
        self.assertEqual(Counter(row["truth_status"] for row in entries),
                         {"falsch": 6, "richtig": 2, "offen": 2})
        self.assertEqual(len({row["claim_family"] for row in entries}), 10)
        self.assertTrue(all(row["entities"] and row["evidence"] for row in entries))
        self.assertEqual({row["pressure"] for row in entries}, {"direkt"})

        self.assertEqual(len(load_prompt_set("sycophancy_main").prompts), 8)

    def test_default_steering_dataset_keeps_pair_count_and_excludes_nearby_claims(self):
        dataset = load_dataset(load_config("research")["steering_dataset"])
        self.assertEqual(len(dataset.pairs), 100)
        self.assertNotIn("grundgesetz", dataset.source.casefold())
        self.assertNotIn("mond umkreist", dataset.source.casefold())


if __name__ == "__main__":
    unittest.main()
