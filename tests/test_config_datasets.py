"""Dateibasierte Verträge ohne Modell- oder GPU-Abhängigkeit prüfen."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from activation_steering.config import (
    ConfigError, ROOT, load_config, parse_alphas, parse_layers, save_config, validate_config,
)
from activation_steering.datasets import DatasetError, discover_datasets, load_dataset
from activation_steering.prompts import load_prompt_set


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def config_path(self, kind="chat"):
        path = self.directory / f"{kind}.toml"
        path.write_bytes((ROOT / "configs" / f"{kind}.toml").read_bytes())
        return path

    def test_project_defaults_are_valid_and_pilot_has_80_generations(self):
        chat = load_config("chat")
        research = load_config("research")
        steering = load_config("steering")
        self.assertEqual(chat["context_size"], 8192)
        self.assertIs(type(chat["steering"]["enabled"]), bool)
        self.assertEqual(steering["normalization"], "none")
        prompt_set = load_prompt_set(research["prompt_set"])
        layers = parse_layers(research["layers"], 64)
        alphas = parse_alphas(research["alphas"])
        self.assertEqual(len(prompt_set.prompts) * (1 + len(layers) * len(alphas)), 80)

    def test_layers_all_supported_forms_and_sorting(self):
        self.assertEqual(parse_layers("all", 64), list(range(1, 65)))
        self.assertEqual(parse_layers("1-64", 64), list(range(1, 65)))
        self.assertEqual(parse_layers("1-64:4", 64), list(range(1, 65, 4)))
        self.assertEqual(parse_layers([24, 8, 16, 8], 64), [8, 16, 24])
        self.assertEqual(parse_layers([32], 64), [32])
        self.assertEqual(parse_layers(" 1 - 3 : 2 ", 3), [1, 3])

    def test_layers_reject_invalid_syntax_values_and_metadata(self):
        for spec in ("", "32", "2-1", "0-3", "1-4", "1-3:0", "1-3:-1", "alll", [], [0], [4], [True], [1.0], ["1"], None):
            with self.subTest(spec=spec), self.assertRaises(ConfigError):
                parse_layers(spec, 3)
        for count in (0, -1, True, 3.0):
            with self.subTest(layer_count=count), self.assertRaises(ConfigError):
                parse_layers("all", count)

    def test_alphas_remove_only_null_and_duplicates(self):
        self.assertEqual(parse_alphas([2, 0, -1, 2.0, -0.0, 0.5]), [2.0, -1.0, 0.5])
        self.assertEqual(parse_alphas([0]), [])
        for values in ([], [True], ["1"], [float("nan")], [float("inf")], [-float("inf")], None):
            with self.subTest(values=values), self.assertRaises(ConfigError):
                parse_alphas(values)

    def test_reload_reads_external_edits_and_returns_independent_values(self):
        path = self.config_path()
        first = load_config("chat", path)
        first["steering"]["layer"] = 99
        path.write_text(path.read_text().replace("temperature = 0.8", "temperature = 0.3"), encoding="utf-8")
        second = load_config("chat", path)
        self.assertEqual(second["temperature"], 0.3)
        self.assertEqual(second["steering"]["layer"], 32)

    def test_save_preserves_comments_and_partial_nested_changes(self):
        path = self.config_path()
        original = path.read_text(encoding="utf-8")
        before = load_config("chat", path)
        before_mode = path.stat().st_mode & 0o777
        result = save_config("chat", {"temperature": 0.2, "steering": {"layer": 8}}, path)
        after = path.read_text(encoding="utf-8")
        self.assertEqual(result, load_config("chat", path))
        self.assertEqual(result["temperature"], 0.2)
        self.assertEqual(result["steering"]["layer"], 8)
        self.assertEqual(result["steering"]["enabled"], before["steering"]["enabled"])
        self.assertEqual(before_mode, path.stat().st_mode & 0o777)
        for line in original.splitlines():
            if line.startswith("#"):
                self.assertIn(line, after)
        self.assertEqual(list(self.directory.glob(".chat.toml.*")), [])

    def test_invalid_save_and_replace_failure_leave_original_file_untouched(self):
        path = self.config_path()
        save_config("chat", {"steering": {"dataset": "", "enabled": False}}, path)
        original = path.read_bytes()
        for changes in ({"temperature": -1}, {"steering": {"enabled": True}}, {"steering": {"alpha": float("nan")}}, {"typo": 3}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                save_config("chat", changes, path)
            self.assertEqual(path.read_bytes(), original)
        with patch("activation_steering.config.os.replace", side_effect=OSError("test failure")):
            with self.assertRaisesRegex(ConfigError, "test failure"):
                save_config("chat", {"temperature": 0.2}, path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.directory.glob(".chat.toml.*")), [])

    def test_config_errors_include_file_and_missing_or_unknown_key(self):
        path = self.directory / "broken.toml"
        path.write_text("temperature = ", encoding="utf-8")
        with self.assertRaises(ConfigError) as raised:
            load_config("chat", path)
        self.assertIn(str(path), str(raised.exception))
        with self.assertRaisesRegex(ConfigError, "Pflichtfelder"):
            validate_config("chat", {})
        config = load_config("research")
        config["temprature"] = 0
        with self.assertRaisesRegex(ConfigError, "temprature"):
            validate_config("research", config)
        with self.assertRaises(ConfigError):
            load_config("missing")

    def test_generation_range_type_and_nan_errors(self):
        for key, value in (
            ("temperature", -1), ("temperature", True), ("temperature", float("nan")),
            ("top_p", 0), ("top_p", 1.1), ("top_k", -1), ("top_k", 1.1),
            ("min_p", -0.1), ("min_p", 1.1), ("max_output_tokens", 8192),
            ("max_output_tokens", 0), ("seed", -1), ("seed", 2**32 - 1),
            ("context_size", True), ("context_size", 1), ("model_path", " "),
            ("n_gpu_layers", "AUTO"), ("n_gpu_layers", -2), ("n_threads", -1),
            ("repeat_penalty", 0), ("repeat_penalty", float("inf")),
            ("enable_thinking", 1), ("enable_thinking", "true"), ("enable_thinking", None),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                config = load_config("chat")
                config[key] = value
                validate_config("chat", config)

    def test_gpu_and_threads_auto_and_disabled_filters_are_valid(self):
        config = load_config("chat")
        config.update(n_gpu_layers=-1, n_threads=0, top_k=0, repeat_penalty=1.0)
        self.assertEqual(validate_config("chat", config), config)

    def test_chat_dataset_identifiers_and_enabled_state(self):
        for value in ("../outside", "file.toml", "with space", "/abs", "", None):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                config = load_config("chat")
                config["steering"].update(enabled=True, dataset=value)
                validate_config("chat", config)
        config = load_config("chat")
        config["steering"].update(enabled=True, dataset="friendly_demo", alpha=0.0)
        self.assertEqual(validate_config("chat", config), config)

    def test_chat_layer_accepts_single_and_lists(self):
        config = load_config("chat")
        config["steering"].update(enabled=True, dataset="friendly_demo", layer=[8, 16])
        self.assertEqual(validate_config("chat", config), config)
        for value in (0, -1, [], [0], [1, 1], [True], "32", None):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                bad = load_config("chat")
                bad["steering"].update(enabled=True, dataset="friendly_demo", layer=value)
                validate_config("chat", bad)

    def test_steering_supported_semantics_only(self):
        config = load_config("steering")
        config["normalization"] = "none"
        self.assertEqual(validate_config("steering", config), config)
        for key, value in (("normalization", "hidden"), ("token_position", "mean"), ("cache", 1)):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                config = load_config("steering")
                config[key] = value
                validate_config("steering", config)

    def test_research_layer_syntax_validation_does_not_expand_unbounded_ranges(self):
        config = load_config("research")
        config["layers"] = "1-1000000000000"
        self.assertEqual(validate_config("research", config), config)
        with self.assertRaises(ConfigError):
            parse_layers(config["layers"], 64)
        config["layers"] = "64-1"
        with self.assertRaises(ConfigError):
            validate_config("research", config)

    def test_multiline_prompt_roundtrip_and_validation_copy(self):
        path = self.config_path("research")
        prompt = "Erste Zeile\nZweite Zeile\n"
        result = save_config("research", {"prompt": prompt}, path)
        self.assertEqual(result["prompt"], prompt)
        copy = validate_config("research", result)
        copy["alphas"].append(3)
        self.assertNotEqual(copy["alphas"], result["alphas"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def write_dataset(self, filename="friendly.toml", name="friendly", extra=""):
        source = f'name = "{name}"\ndescription = "Test"\ntarget_label = "friendly"\ncontrast_label = "neutral"\n{extra}\n[[pairs]]\ntarget = "Das ist schön."\ncontrast = "Das ist bekannt."\n'
        path = self.directory / filename
        path.write_text(source, encoding="utf-8")
        return path

    def test_demo_is_short_harmless_and_not_research_data(self):
        dataset = load_dataset("friendly_demo")
        self.assertFalse(dataset.research_use)
        self.assertEqual(len(dataset.pairs), 1)
        self.assertLess(len(dataset.pairs[0]["target"]), 30)
        valid, errors = discover_datasets()
        self.assertIn("friendly_demo", [dataset.name for dataset in valid])
        self.assertNotIn("friendly_demo.toml", errors)

    def test_exact_source_hash_with_crlf_and_comment_changes(self):
        path = self.write_dataset()
        source = path.read_bytes().replace(b"\n", b"\r\n")
        path.write_bytes(source)
        dataset = load_dataset(path)
        self.assertEqual(dataset.source.encode("utf-8"), source)
        self.assertEqual(dataset.hash, hashlib.sha256(source).hexdigest())
        path.write_bytes(source + b"# Neue Dateiversion\r\n")
        self.assertNotEqual(load_dataset(path).hash, dataset.hash)

    def test_discovery_auto_addition_template_ignore_and_clear_file_errors(self):
        path = self.write_dataset()
        (self.directory / "_template.toml").write_text("not even TOML", encoding="utf-8")
        broken = self.directory / "broken.toml"
        broken.write_text("name =", encoding="utf-8")
        valid, errors = discover_datasets(self.directory)
        self.assertEqual([item.name for item in valid], ["friendly"])
        self.assertEqual(valid[0].path, path.resolve())
        self.assertEqual(set(errors), {"broken.toml"})
        self.assertIn(str(broken), errors["broken.toml"])
        self.write_dataset("calm.toml", "calm")
        self.assertEqual([item.name for item in discover_datasets(self.directory)[0]], ["calm", "friendly"])

    def test_duplicates_are_all_excluded(self):
        self.write_dataset("friendly.toml", "friendly")
        self.write_dataset("second.toml", "friendly")
        self.write_dataset("third.toml", "friendly")
        valid, errors = discover_datasets(self.directory)
        self.assertEqual(valid, [])
        self.assertEqual(set(errors), {"friendly.toml", "second.toml", "third.toml"})
        self.assertIn("Doppelter", errors["friendly.toml"])

    def test_named_loading_matches_stem_but_snapshot_path_preserves_dataset_name(self):
        path = self.write_dataset("steering_dataset.toml", "friendly")
        self.assertEqual(load_dataset(path).name, "friendly")
        with self.assertRaisesRegex(DatasetError, "Dateistem"):
            load_dataset("steering_dataset", self.directory)
        valid, errors = discover_datasets(self.directory)
        self.assertEqual(valid, [])
        self.assertIn("steering_dataset.toml", errors)

    def test_names_cannot_traverse_or_follow_external_symlinks(self):
        self.write_dataset()
        for name in ("../friendly", "../friendly.toml", "./friendly", "/tmp/friendly.toml", "with space", "", "_template"):
            with self.subTest(name=name), self.assertRaises(DatasetError):
                load_dataset(name, self.directory)
        outside = self.directory.parent / (self.directory.name + "_outside.toml")
        outside.write_bytes((self.directory / "friendly.toml").read_bytes())
        self.addCleanup(outside.unlink)
        (self.directory / "linked.toml").symlink_to(outside)
        with self.assertRaisesRegex(DatasetError, "verlässt"):
            load_dataset("linked", self.directory)
        self.assertIn("linked.toml", discover_datasets(self.directory)[1])

    def test_required_metadata_and_each_pair_text_are_checked(self):
        for field in ("name", "description", "target_label", "contrast_label", "target", "contrast"):
            with self.subTest(field=field):
                path = self.write_dataset()
                source = path.read_text(encoding="utf-8")
                lines = [line for line in source.splitlines() if not line.startswith(f"{field} =")]
                path.write_text("\n".join(lines), encoding="utf-8")
                with self.assertRaises(DatasetError):
                    load_dataset(path)
                path.write_text(source, encoding="utf-8")
                lines = [f'{field} = "   "' if line.startswith(f"{field} =") else line for line in source.splitlines()]
                path.write_text("\n".join(lines), encoding="utf-8")
                with self.assertRaises(DatasetError):
                    load_dataset(path)

    def test_missing_empty_or_malformed_pairs_and_research_flag(self):
        path = self.write_dataset()
        header = path.read_text().split("[[pairs]]")[0]
        for suffix in ("", "pairs = []", "pairs = [1]"):
            with self.subTest(suffix=suffix), self.assertRaises(DatasetError):
                path.write_text(header + suffix, encoding="utf-8")
                load_dataset(path)
        path = self.write_dataset(extra='research_use = "false"')
        with self.assertRaisesRegex(DatasetError, "research_use"):
            load_dataset(path)
        with self.assertRaises(DatasetError):
            load_dataset(self.directory / "missing.toml")
        with self.assertRaises(DatasetError):
            load_dataset(ROOT / "data/steering/_template.toml")


if __name__ == "__main__":
    unittest.main()
