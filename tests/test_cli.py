"""Terminalflüsse ohne zweite Instanz des großen lokalen Modells."""

from contextlib import nullcontext
from copy import deepcopy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from rich.console import Console
import tomlkit

from activation_steering.config import load_config, save_config
from activation_steering.datasets import load_dataset
from activation_steering.evaluation import evaluate, latest_evaluations
from activation_steering.storage import append_jsonl, atomic_json
from activation_steering.vectors import VectorSet
from cli import chat, common, main, matrix, research, review


class FakeBackend:
    instances = []

    def __init__(self, config):
        self.config = config
        self.model = SimpleNamespace(n_tokens=0)
        self.layer_count = 64
        self.dimension = 2
        self.metadata = {"model_name": "test", "model_path": "test.gguf", "quantization": "test"}
        self.load_time = 0.1
        self.closed = False
        self.resets = 0
        self.steering_resets = 0
        self.calls = []
        self.__class__.instances.append(self)

    def __enter__(self):
        if any(not instance.closed for instance in self.instances if instance is not self):
            raise AssertionError("Zwei Modelle gleichzeitig geladen")
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True

    def reset(self):
        self.model.n_tokens = 0
        self.resets += 1

    def reset_steering(self):
        self.steering_resets += 1

    def format_prompt(self, messages, enable_thinking=True):
        return "<think>" if enable_thinking else "<think>\n\n</think>"

    def tokens(self, prompt):
        return [1, 2]

    def generate(self, messages, config, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(config), kwargs))
        if kwargs.get("on_piece"):
            kwargs["on_piece"]("Gedanke</think>Vier. [bold]")
        self.model.n_tokens = 5
        return {"response": "Gedanke</think>Vier. [bold]", "output_tokens": 3,
                "prompt_tokens": 2, "ttft": 0.1, "generation_time": 0.2,
                "total_time": 0.3, "tokens_per_second": 10, "seed": config["seed"],
                "finish_reason": "length", "output_token_ids": [1, 2, 3]}

    def alpha_zero_control(self, config, vector, layer):
        self.control = (config, vector, layer)
        return {"passed": True, "layer": layer}


class CliTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.console = Console(file=self.output, width=80, height=25, color_system=None)
        self.capture = patch.object(common, "console", self.console)
        self.capture.start()
        self.matrix_console = patch.object(matrix, "console", self.console)
        self.matrix_console.start()
        self.chat_console = patch.object(chat, "console", self.console)
        self.chat_console.start()
        FakeBackend.instances = []

    def tearDown(self):
        self.capture.stop()
        self.matrix_console.stop()
        self.chat_console.stop()

    def fixture(self, directory, *, prompts=1):
        run_dir = Path(directory)
        atomic_json(run_dir / "metadata.json", {
            "run_id": "test", "status": "interrupted", "total": prompts * 385,
            "completed": 3, "artifact_hashes": {}, "backend_fingerprint": {},
            "layers": list(range(1, 65)), "alphas": [-2, -1, -.5, .5, 1, 2],
        })
        snapshot = {"prompt": "Bonjour [bold]", "resolved": {
            "layers": list(range(1, 65)), "alphas": [-2, -1, -.5, .5, 1, 2],
            "prompts": [f"Prompt {index}" for index in range(prompts)],
        }}
        (run_dir / "config.toml").write_text(tomlkit.dumps(snapshot), encoding="utf-8")
        atomic_json(run_dir / "steering_metadata.json", {"target_label": "friendly",
                    "contrast_label": "neutral", "dataset_description": "Demo"})
        (run_dir / "evaluations.jsonl").write_text("")
        rows = [
            {"result_id": "b", "type": "baseline", "layer": None, "alpha": 0,
             "prompt_id": "p1", "prompt": "Bonjour [bold]", "response": "Baseline [bold]"},
            {"result_id": "s1", "type": "steering", "layer": 1, "alpha": -2,
             "prompt_id": "p1", "prompt": "Bonjour [bold]", "response": "Antwort [bold]"},
            {"result_id": "s2", "type": "steering", "layer": 64, "alpha": 2,
             "prompt_id": "p1", "prompt": "Bonjour", "response": "Antwort zwei"},
        ]
        for row in rows:
            row.update(output_tokens=10, prompt_tokens=3, ttft=.2, tokens_per_second=5,
                       generation_time=2, total_time=2.2, repeat_id=0, seed=42,
                       finish_reason="length", created_at="2026-10-07T00:00:00+00:00",
                       output_token_ids=list(range(10)))
            append_jsonl(run_dir / "results.jsonl", row)
        return run_dir, rows

    def test_main_menu_does_not_load_model_and_eof_exits(self):
        with patch("builtins.input", side_effect=["invalid", "4"]), patch("activation_steering.backend.Backend") as backend:
            self.assertEqual(main.main(), 0)
            backend.assert_not_called()
        self.assertIn("CHAPPiE Activation Steering", self.output.getvalue())
        with patch("builtins.input", side_effect=EOFError):
            self.assertEqual(main.main(), 0)

    def test_main_menu_model_entry_switches_without_loading_model(self):
        with patch("builtins.input", side_effect=["3", "4"]), \
             patch.object(main, "model_menu") as model_menu, \
             patch("activation_steering.backend.Backend") as backend:
            self.assertEqual(main.main(), 0)
        model_menu.assert_called_once()
        backend.assert_not_called()

    def test_choose_model_and_model_menu_update_both_configs(self):
        with TemporaryDirectory() as directory:
            models = Path(directory) / "models"
            models.mkdir()
            (models / "one.gguf").write_bytes(b"1")
            (models / "two.gguf").write_bytes(b"2")
            with patch.object(common, "MODELS_DIR", models):
                self.assertEqual([path.name for path in common.discover_models()], ["one.gguf", "two.gguf"])
                with patch("builtins.input", side_effect=["2"]):
                    self.assertEqual(common.choose_model(), str(models / "two.gguf"))
                with patch("builtins.input", side_effect=["q"]):
                    self.assertIsNone(common.choose_model())
            empty = Path(directory) / "empty"
            empty.mkdir()
            with patch.object(common, "MODELS_DIR", empty), patch("builtins.input") as prompts:
                self.assertIsNone(common.choose_model())
                prompts.assert_not_called()
            saved = []
            with patch.object(common, "MODELS_DIR", models), \
                 patch.object(common, "load_config", return_value=load_config("chat")), \
                 patch.object(common, "save_config", side_effect=lambda kind, changes: saved.append((kind, changes))), \
                 patch("builtins.input", side_effect=["2"]):
                common.model_menu()
            target = str(models / "two.gguf")
            self.assertEqual(saved, [("chat", {"model_path": target}), ("research", {"model_path": target})])
            self.assertIn("two.gguf", self.output.getvalue())

    def test_settings_menu_model_path_uses_models_directory(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chat.toml"
            path.write_text(Path("configs/chat.toml").read_text(encoding="utf-8"), encoding="utf-8")
            models = Path(directory) / "models"
            models.mkdir()
            (models / "other.gguf").write_bytes(b"x")

            def load(kind):
                return load_config(kind, path if kind == "chat" else None)

            def save(kind, changes):
                return save_config(kind, changes, path if kind == "chat" else None)

            with patch.object(common, "MODELS_DIR", models), \
                 patch.object(common, "load_config", side_effect=load), \
                 patch.object(common, "save_config", side_effect=save), \
                 patch("builtins.input", side_effect=["1", "1", "q"]):
                common.settings_menu("chat")
            self.assertEqual(load_config("chat", path)["model_path"], str(models / "other.gguf"))

    def test_chat_streaming_history_and_hot_reload_only_on_new(self):
        original = load_config("chat")
        original["steering"] = {**original["steering"], "enabled": False}
        updated = {**original, "temperature": 0.2}
        with patch("builtins.input", side_effect=["Hallo", "Nochmal", "/status", "/new", "Neu", "/exit"]), \
             patch.object(chat, "load_config", side_effect=[original, updated]) as configs, \
             patch("activation_steering.backend.Backend", FakeBackend):
            chat.chat_session()
        backend = FakeBackend.instances[0]
        self.assertEqual(configs.call_count, 2)
        self.assertEqual(len(backend.calls[1][0]), 3)
        self.assertEqual(backend.calls[1][0][1]["reasoning_content"], "Gedanke")
        self.assertEqual(backend.calls[1][1]["temperature"], original["temperature"])
        self.assertEqual(backend.calls[2][1]["temperature"], .2)
        self.assertEqual(len(backend.calls[2][0]), 1)
        self.assertEqual(backend.steering_resets, 3)
        self.assertTrue(backend.closed)
        self.assertIn("[bold]", self.output.getvalue())
        self.assertIn("TTFT", self.output.getvalue())

    def test_chat_context_change_closes_old_model_before_reload(self):
        original = load_config("chat")
        original["steering"] = {**original["steering"], "enabled": False}
        updated = {**original, "context_size": 4096}
        with patch("builtins.input", side_effect=["/new", "/exit"]), \
             patch.object(chat, "load_config", side_effect=[original, updated]), \
             patch("activation_steering.backend.Backend", FakeBackend):
            chat.chat_session()
        self.assertEqual(len(FakeBackend.instances), 2)
        self.assertTrue(all(instance.closed for instance in FakeBackend.instances))

    def test_chat_context_exhaustion_does_not_accept_message(self):
        config = load_config("chat")
        config["steering"] = {**config["steering"], "enabled": False}
        with patch("builtins.input", side_effect=["Zu lang", "/exit"]), \
             patch.object(chat, "load_config", return_value=config), \
             patch("activation_steering.backend.Backend", FakeBackend), \
             patch.object(FakeBackend, "tokens", return_value=list(range(8190))):
            chat.chat_session()
        self.assertEqual(FakeBackend.instances[0].calls, [])
        self.assertIn("Nachricht nicht übernommen", self.output.getvalue())

    def test_assistant_thinking_and_plain_responses(self):
        self.assertEqual(chat.assistant_message("Plain", "prompt"), {"role": "assistant", "content": "Plain"})
        self.assertEqual(chat.assistant_message("thought", "prompt<think>")["reasoning_content"], "thought")
        self.assertEqual(chat.assistant_message("<think>thought</think>answer", "prompt")["content"], "answer")

    def test_chat_thinking_toggle_updates_session_and_saved_config(self):
        config = load_config("chat")
        config["enable_thinking"] = True
        config["steering"] = {"enabled": False, "dataset": "", "layer": 32, "alpha": 1.0}
        saved = {}
        with patch("builtins.input", side_effect=["/thinking", "/thinking off", "Hallo", "/exit"]), \
             patch.object(chat, "load_config", return_value=config), \
             patch.object(chat, "save_config", side_effect=lambda kind, changes: saved.update(changes)), \
             patch("activation_steering.backend.Backend", FakeBackend):
            chat.chat_session()
        self.assertEqual(saved, {"enable_thinking": False})
        backend = FakeBackend.instances[0]
        self.assertEqual(backend.calls[0][1]["enable_thinking"], False)
        self.assertIn("Thinking", self.output.getvalue())
        self.assertIn("Steering: AUS", self.output.getvalue())

    def test_chat_layer_input_accepts_single_and_lists(self):
        self.assertEqual(common.parse_layers_input("32"), 32)
        self.assertEqual(common.parse_layers_input("8,16,24"), [8, 16, 24])
        self.assertEqual(common.parse_layers_input("[8, 16]"), [8, 16])
        self.assertEqual(common.parse_layers_input("8 16 24"), [8, 16, 24])
        for raw in ("", "0", "-1", "1,1", "a", "[]"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                common.parse_layers_input(raw)
        for raw, expected in (("all", "all"), ("1-64:4", "1-64:4"), ("8, 16", [8, 16]), ("32", [32])):
            with self.subTest(raw=raw):
                self.assertEqual(common.parse_layers_spec(raw), expected)

    def test_chat_multi_layer_generates_one_answer_per_layer(self):
        from types import SimpleNamespace
        config = load_config("chat")
        config["enable_thinking"] = False
        config["steering"] = {"enabled": True, "dataset": "friendly_demo", "layer": [8, 16], "alpha": 1.0}
        vectors = SimpleNamespace(vectors={8: [1, 0], 16: [0, 1]})
        with patch("builtins.input", side_effect=["Hallo", "/exit"]), \
             patch.object(chat, "load_config", return_value=config), \
             patch("activation_steering.backend.Backend", FakeBackend), \
             patch("activation_steering.vectors.build_vectors", return_value=vectors):
            chat.chat_session()
        backend = FakeBackend.instances[0]
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([call[2]["layer"] for call in backend.calls], [8, 16])
        self.assertIn("Layer 8", self.output.getvalue())
        self.assertIn("Layer 16", self.output.getvalue())

    def test_mixed_gpu_config_values_keep_types(self):
        self.assertEqual(common._parse_value("0", "auto", "n_gpu_layers"), 0)
        self.assertEqual(common._parse_value("-1", "auto", "n_gpu_layers"), -1)
        self.assertEqual(common._parse_value("auto", 0, "n_gpu_layers"), "auto")
        self.assertEqual(common._parse_value("auto", "auto", "n_gpu_layers"), "auto")
        self.assertEqual(common._parse_value("123", "", "prompt"), "123")
        self.assertIs(common._parse_value("False", True, "enable_thinking"), False)
        self.assertIs(common._parse_value("an", False, "enable_thinking"), True)
        self.assertEqual(common._parse_value("1, 2, 3", [8, 16], "alphas"), [1, 2, 3])
        self.assertEqual(common._parse_value("[8, 16]", [32], "layers"), [8, 16])

    def test_settings_menu_preserves_comments_and_gpu_type(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chat.toml"
            source = Path("configs/chat.toml").read_text(encoding="utf-8")
            path.write_text(source, encoding="utf-8")
            keys = [key for _, _, section in common.SETTINGS_SECTIONS["chat"] for key in section]
            index = keys.index("n_gpu_layers") + 1

            def load(kind):
                return load_config(kind, path if kind == "chat" else None)

            def save(kind, changes):
                return save_config(kind, changes, path if kind == "chat" else None)

            with patch.object(common, "load_config", side_effect=load), \
                 patch.object(common, "save_config", side_effect=save), \
                 patch("builtins.input", side_effect=[str(index), "0", "q"]):
                common.settings_menu("chat")
            self.assertEqual(load_config("chat", path)["n_gpu_layers"], 0)
            self.assertIn("# Neue Chats", path.read_text(encoding="utf-8"))

    def test_settings_menu_enabling_steering_prompts_for_dataset(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chat.toml"
            path.write_text(Path("configs/chat.toml").read_text(encoding="utf-8"), encoding="utf-8")
            save_config("chat", {"steering": {"dataset": "", "enabled": False}}, path)

            def load(kind):
                return load_config(kind, path if kind == "chat" else None)

            def save(kind, changes):
                return save_config(kind, changes, path if kind == "chat" else None)

            with patch.object(common, "load_config", side_effect=load), \
                 patch.object(common, "save_config", side_effect=save), \
                 patch("builtins.input", side_effect=["13", "true", "1", "q"]):
                common.settings_menu("chat")
            steering = load_config("chat", path)["steering"]
            self.assertTrue(steering["enabled"])
            self.assertEqual(steering["dataset"], "anger")

    def test_settings_menu_keeps_value_on_empty_input(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chat.toml"
            path.write_text(Path("configs/chat.toml").read_text(encoding="utf-8"), encoding="utf-8")
            save_config("chat", {"steering": {"dataset": "", "enabled": False}}, path)

            def load(kind):
                return load_config(kind, path if kind == "chat" else None)

            def save(kind, changes):
                return save_config(kind, changes, path if kind == "chat" else None)

            with patch.object(common, "load_config", side_effect=load), \
                 patch.object(common, "save_config", side_effect=save), \
                 patch("builtins.input", side_effect=["13", "", "q"]):
                common.settings_menu("chat")
            self.assertFalse(load_config("chat", path)["steering"]["enabled"])
            self.assertIn("Unverändert.", self.output.getvalue())

    def test_alpha_zero_control_selects_native_or_first_layer(self):
        for layer_text, expected in (("", 32), ("1", 1)):
            FakeBackend.instances = []
            with patch("builtins.input", side_effect=[layer_text, "y"]), \
                 patch.object(research, "experiment_lock", return_value=nullcontext()), \
                 patch("activation_steering.backend.Backend", FakeBackend), \
                 patch("activation_steering.vectors.build_vectors", return_value=SimpleNamespace(vectors={1: [1, 0], 32: [1, 0]})):
                research.alpha_zero_control()
            config, _, layer = FakeBackend.instances[-1].control
            self.assertEqual(layer, expected)
            self.assertEqual(config["temperature"], 0)
            self.assertEqual(config["max_output_tokens"], 16)
            self.assertIn("passed", self.output.getvalue())

    def test_empty_multiline_prompt_and_cancel(self):
        with patch("builtins.input", side_effect=["/done", "erste Zeile", "zweite Zeile", "/done"]):
            self.assertEqual(research.multiline_prompt(), "erste Zeile\nzweite Zeile")
        with patch("builtins.input", side_effect=["/cancel"]):
            self.assertIsNone(research.multiline_prompt())

    def test_preview_has_385_generations_and_all_settings(self):
        config = load_config("research")
        config.update(prompt="Harmloser Prompt", steering_dataset="friendly_demo")
        backend = FakeBackend(config)
        plan = research.preview_run(config, load_dataset("friendly_demo"), backend, load_config("steering"))
        self.assertEqual(plan["total"], 385)
        self.assertEqual(plan["baseline"], 1)
        self.assertIn("research_use = false", self.output.getvalue())
        self.assertIn("max_output_tokens", self.output.getvalue())
        self.assertIn("Gesamt: 385", self.output.getvalue())

    def test_declined_preview_creates_no_run_or_vectors(self):
        config = load_config("research")
        config.update(prompt="Harmloser Prompt", steering_dataset="friendly_demo")
        lock = SimpleNamespace(update_run=lambda value: None)
        with patch.object(research, "load_config", side_effect=[config, load_config("steering")]), \
             patch.object(research, "experiment_lock", return_value=nullcontext(lock)), \
             patch("activation_steering.backend.Backend", FakeBackend), \
             patch("activation_steering.vectors.build_vectors") as vectors, \
             patch.object(research, "create_run") as create, \
             patch("builtins.input", return_value="n"):
            research.start_run()
        create.assert_not_called()
        vectors.assert_not_called()
        self.assertTrue(FakeBackend.instances[0].closed)

    def test_confirmed_research_flow_locks_snapshots_and_executes_once(self):
        import activation_steering.runs as runs
        config = load_config("research")
        config.update(prompt="Harmloser Prompt", steering_dataset="friendly_demo", layers=[1], alphas=[0, 1])
        dataset = load_dataset("friendly_demo")
        vectors = VectorSet({1: np.array([1, 0], dtype=np.float32)},
                            {"dataset_hash": dataset.hash, "layers": [1], "vector_dimension": 2})
        with TemporaryDirectory() as directory:
            with patch.object(runs, "RUNS_DIR", Path(directory)), \
                 patch.object(research, "load_config", side_effect=[config, load_config("steering")]), \
                 patch("activation_steering.backend.Backend", FakeBackend), \
                 patch("activation_steering.vectors.build_vectors", return_value=vectors), \
                 patch("builtins.input", return_value="y"):
                research.start_run()
            paths = runs.discover_runs(directory)
            self.assertEqual(len(paths), 1)
            self.assertEqual(runs.read_metadata(paths[0])["status"], "complete")
            results = runs.read_results(paths[0])
            self.assertEqual([result["type"] for result in results], ["baseline", "steering"])
            self.assertFalse((Path(directory) / ".experiment.lock").exists())
            self.assertTrue(FakeBackend.instances[0].closed)
            self.assertIsNone(FakeBackend.instances[0].calls[0][2]["vector"])

    def test_review_filters_and_baseline_applicability(self):
        with TemporaryDirectory() as directory:
            _, rows = self.fixture(directory)
            choices = review.review_candidates(rows, {}, metric="steering_effect")
            self.assertEqual([row["result_id"] for row in choices], ["s1", "s2"])
            self.assertEqual(len(review.review_candidates(rows, {}, layer=64)), 1)
            self.assertEqual(len(review.review_candidates(rows, {}, alpha=-2)), 1)
            reviewed = {("s1", "refusal"): {"score": 0}}
            self.assertEqual([row["result_id"] for row in review.review_candidates(rows, reviewed, reviewed=True)], ["s1"])

    def test_review_partial_scores_invalid_input_and_raw_integrity(self):
        with TemporaryDirectory() as directory:
            run_dir, _ = self.fixture(directory)
            before = (run_dir / "results.jsonl").read_bytes()
            with patch("builtins.input", side_effect=["9", "2", "q"]):
                review.review_answers(run_dir, metric="quality")
            self.assertEqual(latest_evaluations(run_dir)[("b", "quality")]["score"], 2)
            self.assertEqual(before, (run_dir / "results.jsonl").read_bytes())
            self.assertIn("Bewertung nicht gespeichert", self.output.getvalue())
            self.assertIn("Baseline [bold]", self.output.getvalue())

    def test_review_skip_and_blank_leave_scores_open(self):
        with TemporaryDirectory() as directory:
            run_dir, _ = self.fixture(directory)
            with patch("builtins.input", side_effect=["", "s", "q"]):
                review.review_answers(run_dir, metric="quality")
            self.assertEqual(latest_evaluations(run_dir), {})

    def test_next_open_review_skips_existing_metrics(self):
        with TemporaryDirectory() as directory:
            run_dir, _ = self.fixture(directory)
            evaluate(run_dir, "b", "refusal", 0)
            prompts = []

            def answer(prompt):
                prompts.append(prompt)
                return "q"

            with patch("builtins.input", side_effect=answer), patch.object(review, "show_rubric", wraps=review.show_rubric) as rubric:
                review.review_answers(run_dir)
            self.assertEqual(rubric.call_args.args[0], "quality")
            self.assertEqual(latest_evaluations(run_dir)[("b", "refusal")]["score"], 0)

    def test_matrix_metrics_partial_reviews_and_two_axis_pagination(self):
        self.assertEqual(len(matrix.MATRIX_METRICS), 8)
        self.assertEqual(matrix.cell_value([{"result_id": "one"}], "quality", {}, 2)[0], "?\n(0/2)")
        self.assertEqual(matrix.cell_value([{"result_id": "one"}], "quality", {("one", "quality"): {"score": 3}}, 2)[0], "3\n(1/2)")
        with TemporaryDirectory() as directory:
            run_dir, _ = self.fixture(directory)
            evaluate(run_dir, "b", "quality", 3)
            self.console.width = 35
            first = matrix.render_matrix(run_dir, "quality")
            last = matrix.render_matrix(run_dir, "output_tokens", 999, 999)
            self.assertGreater(first[2], 1)
            self.assertGreater(first[3], 1)
            self.assertEqual(last[0], last[2] - 1)
            self.assertEqual(last[1], last[3] - 1)
            output = self.output.getvalue()
            self.assertIn("64", output)
            self.assertIn("Gemeinsame Baseline", output)
            self.assertNotIn("…", output)

    def test_matrix_layer_and_metric_navigation(self):
        with TemporaryDirectory() as directory:
            run_dir, _ = self.fixture(directory)
            with patch("builtins.input", side_effect=["n", "a", "m", "5", "l", "64", "q", "q"]):
                matrix.matrix_view(run_dir)
            self.assertIn("Output Tokens", self.output.getvalue())
            self.assertIn("Layer 64", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
