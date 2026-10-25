"""Prompt-set validation without loading a model."""

from pathlib import Path
import tempfile
import unittest

from activation_steering.prompts import PromptSetError, discover_prompt_sets, load_prompt_set


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


if __name__ == "__main__":
    unittest.main()
