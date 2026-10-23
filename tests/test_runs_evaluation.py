"""Deterministic storage and runner checks; never load a model or invent scores."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import unittest

import numpy as np

from activation_steering.datasets import load_dataset
from activation_steering.evaluation import METRICS, evaluate, latest_evaluations, summarize
from activation_steering.runs import (
    ExperimentLock, LockError, RunError, create_run, discover_runs, execute_run,
    plan_run, read_metadata, read_results, read_snapshot_config, recover_lock,
)
from activation_steering.storage import StorageError, append_jsonl, atomic_json, read_jsonl
from activation_steering.vectors import VectorSet


class FakeBackend:
    """A deterministic runner fixture, explicitly not an inference substitute."""

    def __init__(self, config, layers=2):
        self.config = copy.deepcopy(config)
        self.layer_count = layers
        self.dimension = 3
        self.metadata = {
            "model_hash": "fixture-model", "chat_template": "fixture-template", "bridge_version": "fixture-bridge",
            "backend_library_hashes": {"fixture.so": "fixture-library"}, "context_size": config["context_size"],
            "llama_cpp_python_version": "fixture", "n_gpu_layers": 0, "n_threads": 1,
        }
        self.calls = []
        self.fail_at = None
        self.failure = KeyboardInterrupt
        self.steering_active = False
        self.resets = 0

    def reset_steering(self):
        self.steering_active = False
        self.resets += 1

    def generate(self, messages, config, vector=None, layer=None, alpha=0, on_piece=None):
        self.steering_active = vector is not None
        try:
            if self.fail_at == len(self.calls):
                self.fail_at = None
                raise self.failure("fixture failure")
            self.calls.append({"messages": copy.deepcopy(messages), "config": copy.deepcopy(config), "layer": layer,
                               "alpha": alpha, "vector": None if vector is None else vector.copy()})
            return {"response": f"fixture output {layer} {alpha}", "prompt_tokens": 5, "output_tokens": 2,
                    "ttft": 0.1, "generation_time": 0.2, "total_time": 0.3, "tokens_per_second": 5.0,
                    "seed": config["seed"], "output_token_ids": [1, 2], "finish_reason": "eog"}
        finally:
            self.steering_active = False


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runs_dir = self.root / "runs"
        self.config = {
            "prompt": "Give a friendly greeting.", "layers": [1, 2], "alphas": [-1.0, 0.0, 1.0],
            "steering_dataset": "friendly_demo", "context_size": 8192, "temperature": 0.0, "top_k": 40,
            "top_p": 1.0, "min_p": 0.05, "max_output_tokens": 8, "seed": 42, "repeat_penalty": 1.1,
            "model_path": "fixture.gguf", "n_gpu_layers": 0, "n_threads": 1,
        }
        self.backend = FakeBackend(self.config)
        path = self.root / "friendly_demo.toml"
        path.write_bytes(b'name = "friendly_demo"\r\ndescription = "Harmless fixture."\r\ntarget_label = "friendly"\r\ncontrast_label = "neutral"\r\nresearch_use = false\r\n[[pairs]]\r\ntarget = "Hello, friend."\r\ncontrast = "Hello."\r\n')
        self.dataset = load_dataset(path)
        self.vectors = VectorSet({1: np.array([1, 0, 0], np.float32), 2: np.array([0, 1, 0], np.float32)}, {
            "dataset_hash": self.dataset.hash, "layers": [1, 2], "vector_dimension": 3,
            "normalization": "unit_l2", "raw_norms": {"1": 1.0, "2": 1.0},
            "token_position": "last_non_padding", "extraction_method": "fixture-only", "created_at": "fixture-time",
        })

    def tearDown(self):
        self.temporary.cleanup()

    def create(self):
        return create_run(self.config, self.dataset, self.backend, self.vectors, self.runs_dir)

    def complete(self):
        path = self.create()
        self.assertEqual(execute_run(path, self.backend), "complete")
        return path

    def test_preview_385_and_single_alpha_zero_baseline(self):
        plan = plan_run({**self.config, "layers": "all", "alphas": [-2, -1, -.5, 0, .5, 1, 2]}, 64)
        self.assertEqual((plan["baseline"], plan["steering"], plan["total"]), (1, 384, 385))
        self.assertNotIn(0, plan["alphas"])
        self.assertEqual(plan_run({**self.config, "alphas": [0, 0.0]}, 2)["total"], 1)
        with self.assertRaises(RunError):
            plan_run({**self.config, "prompt": " "}, 2)

    def test_exact_snapshots_and_unique_history(self):
        first, second = self.create(), self.create()
        self.assertNotEqual(first, second)
        self.assertEqual(discover_runs(self.runs_dir), [second, first])
        self.assertEqual((first / "steering_dataset.toml").read_bytes(), self.dataset.path.read_bytes())
        self.assertEqual(read_snapshot_config(first)["resolved"]["layers"], [1, 2])
        self.assertEqual(read_metadata(first)["dataset_hash"], self.dataset.hash)
        for name in ("config.toml", "metadata.json", "steering_dataset.toml", "steering_metadata.json", "vectors.npz", "results.jsonl", "evaluations.jsonl", "summary.json", "run.log"):
            self.assertTrue((first / name).is_file(), name)
        self.assertEqual(read_results(first), [])
        self.assertEqual(latest_evaluations(first), {})
        self.assertEqual(summarize(first)["completed"], 0)

    def test_execution_freezes_config_baseline_isolated_and_progress_durable(self):
        path = self.create()
        self.config["prompt"] = "Modified outside the run."
        self.config["seed"] = 111
        observed = []

        def progress(value):
            observed.append(dict(value))
            self.assertEqual(len(read_results(path)), value["completed"])

        self.assertEqual(execute_run(path, self.backend, progress), "complete")
        results = read_results(path)
        self.assertEqual(len(results), 5)
        self.assertEqual(sum(result["type"] == "baseline" for result in results), 1)
        self.assertIsNone(self.backend.calls[0]["vector"])
        self.assertIsNone(results[0]["layer"])
        self.assertEqual(results[0]["alpha"], 0.0)
        self.assertFalse(self.backend.steering_active)
        self.assertEqual({call["config"]["seed"] for call in self.backend.calls}, {42})
        self.assertEqual({call["messages"][0]["content"] for call in self.backend.calls}, {"Give a friendly greeting."})
        self.assertEqual(observed[-1]["completed"], 5)
        self.assertEqual(observed[-1]["eta"], 0.0)
        self.assertFalse((self.runs_dir / ".experiment.lock").exists())
        self.assertEqual(summarize(path)["review_counts"]["steering_effect"], {"reviewed": 0, "total": 4})

    def test_multiple_prompts_get_one_baseline_each(self):
        self.config["prompts"] = ["One.", "Two."]
        path = self.complete()
        results = read_results(path)
        self.assertEqual(len(results), 10)
        baselines = [result for result in results if result["type"] == "baseline"]
        self.assertEqual([result["prompt_id"] for result in baselines], ["p0001", "p0002"])

    def test_interrupt_resume_only_missing_combinations(self):
        path = self.create()
        self.backend.fail_at = 2
        self.assertEqual(execute_run(path, self.backend), "interrupted")
        preserved = (path / "results.jsonl").read_bytes()
        self.assertEqual(len(read_results(path)), 2)
        self.assertEqual(read_metadata(path)["status"], "interrupted")
        self.assertFalse(self.backend.steering_active)
        self.assertFalse((self.runs_dir / ".experiment.lock").exists())
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertEqual(len(self.backend.calls), 5)
        self.assertTrue((path / "results.jsonl").read_bytes().startswith(preserved))
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertEqual(len(self.backend.calls), 5)

    def test_failure_is_resumable(self):
        path = self.create()
        self.backend.fail_at = 1
        self.backend.failure = ValueError
        with self.assertRaisesRegex(ValueError, "fixture failure"):
            execute_run(path, self.backend)
        self.assertEqual(read_metadata(path)["status"], "failed")
        self.assertEqual(len(read_results(path)), 1)
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertEqual(len(self.backend.calls), 5)

    def test_progress_interrupt_resets_existing_steering(self):
        path = self.create()
        self.backend.steering_active = True

        def stop(_progress):
            raise KeyboardInterrupt

        self.assertEqual(execute_run(path, self.backend, stop), "interrupted")
        self.assertFalse(self.backend.steering_active)
        self.assertEqual(self.backend.resets, 1)
        self.assertEqual(read_results(path), [])

    def test_resume_repairs_torn_evaluation_tail_without_new_generations(self):
        path = self.complete()
        steering = read_results(path)[1]
        evaluate(path, steering["result_id"], "quality", 2)
        with (path / "evaluations.jsonl").open("ab") as output:
            output.write(b'{"result_id":')
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertEqual(len(self.backend.calls), 5)
        self.assertEqual(latest_evaluations(path)[(steering["result_id"], "quality")]["score"], 2)
        self.assertEqual(len(list(path.glob("evaluations.jsonl.corrupt-*.bin"))), 1)

    def test_incomplete_or_invalid_complete_json_record_never_counts_as_finished(self):
        path = self.complete()
        result_path = path / "results.jsonl"
        original = result_path.read_bytes()
        complete_record = read_results(path)[0]
        bad_records = [{"result_id": complete_record["result_id"]},
                       {**complete_record, "response": None}, {**complete_record, "output_tokens": -1},
                       {**complete_record, "total_time": -1}, {**complete_record, "prompt_tokens": True},
                       {**complete_record, "output_token_ids": []}, {**complete_record, "seed": None},
                       {**complete_record, "ttft": "0.1"}, {**complete_record, "ttft": None},
                       {**complete_record, "alpha": 10**400}]
        for record in bad_records:
            with self.subTest(record=record):
                result_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                before = result_path.read_bytes()
                with self.assertRaises(StorageError):
                    execute_run(path, self.backend)
                self.assertEqual(result_path.read_bytes(), before)
                self.assertEqual(len(self.backend.calls), 5)
        result_path.write_bytes(original)

    def test_malformed_backend_generation_is_not_persisted(self):
        path = self.create()
        original_generate = self.backend.generate

        def malformed(*args, **kwargs):
            return {**original_generate(*args, **kwargs), "seed": 999}

        self.backend.generate = malformed
        with self.assertRaisesRegex(RunError, "eingefrorenen Seed"):
            execute_run(path, self.backend)
        self.assertEqual(read_results(path), [])
        self.assertEqual(read_metadata(path)["status"], "failed")

    def test_stored_generation_with_changed_seed_never_skipped(self):
        path = self.complete()
        records = read_results(path)
        records[0]["seed"] = 999
        (path / "results.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        with self.assertRaisesRegex(RunError, "Gespeicherte Generation"):
            execute_run(path, self.backend)
        self.assertEqual(len(self.backend.calls), 5)

    def test_snapshot_changes_and_backend_drift_rejected(self):
        for artifact in ("config.toml", "steering_dataset.toml", "steering_metadata.json", "vectors.npz"):
            with self.subTest(artifact=artifact):
                path = self.create()
                with (path / artifact).open("ab") as output:
                    output.write(b"altered")
                with self.assertRaisesRegex(RunError, "verändert"):
                    execute_run(path, self.backend)
        path = self.create()
        for key in ("model_hash", "chat_template", "bridge_version", "backend_library_hashes", "n_gpu_layers", "n_threads"):
            with self.subTest(key=key):
                original = self.backend.metadata[key]
                self.backend.metadata[key] = "modified"
                with self.assertRaisesRegex(RunError, "unterscheidet"):
                    execute_run(path, self.backend)
                self.backend.metadata[key] = original
        self.assertEqual(self.backend.calls, [])

    def test_other_dataset_vectors_rejected_before_run_creation(self):
        self.vectors.metadata["dataset_hash"] = "wrong"
        with self.assertRaisesRegex(RunError, "ausgewählten Dataset"):
            self.create()
        self.assertFalse(self.runs_dir.exists())

    def test_actual_offload_and_threads_frozen_in_snapshot(self):
        self.config["n_gpu_layers"] = "auto"
        self.config["n_threads"] = 0
        path = self.create()
        snapshot = read_snapshot_config(path)
        self.assertEqual(snapshot["n_gpu_layers"], 0)
        self.assertEqual(snapshot["n_threads"], 1)
        self.assertEqual(read_metadata(path)["requested_model_settings"], {"n_gpu_layers": "auto", "n_threads": 0})

    def test_preacquired_lock_and_exclusive_active_lock(self):
        path = self.create()
        with ExperimentLock(runs_dir=self.runs_dir) as owner:
            with self.assertRaises(LockError):
                with ExperimentLock(runs_dir=self.runs_dir, recover=True):
                    pass
            with self.assertRaises(LockError):
                recover_lock(self.runs_dir)
            self.assertEqual(execute_run(path, self.backend, lock=owner), "complete")
            self.assertTrue(owner.acquired)
            self.assertTrue((self.runs_dir / ".experiment.lock").exists())
        self.assertFalse((self.runs_dir / ".experiment.lock").exists())

    def test_stale_lock_requires_explicit_recovery_and_keeps_record(self):
        self.runs_dir.mkdir()
        lock_path = self.runs_dir / ".experiment.lock"
        atomic_json(lock_path, {"pid": 999999999, "hostname": socket.gethostname(), "run_id": "crashed", "start_time": "fixture"})
        original = lock_path.read_bytes()
        with self.assertRaisesRegex(LockError, "Recovery"):
            with ExperimentLock(runs_dir=self.runs_dir):
                pass
        self.assertEqual(lock_path.read_bytes(), original)
        with ExperimentLock(runs_dir=self.runs_dir, recover=True):
            self.assertTrue(lock_path.exists())
        backups = list(self.runs_dir.glob(".experiment.lock.recovered-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_foreign_host_lock_never_recovered(self):
        self.runs_dir.mkdir()
        atomic_json(self.runs_dir / ".experiment.lock", {"pid": 999999999, "hostname": "another-server", "run_id": "remote"})
        with self.assertRaises(LockError):
            recover_lock(self.runs_dir)

    def test_torn_tail_recovered_atomically_preserving_all_bytes(self):
        path = self.create()
        self.backend.fail_at = 1
        self.assertEqual(execute_run(path, self.backend), "interrupted")
        result_path = path / "results.jsonl"
        valid_prefix = result_path.read_bytes()
        tail = b'{"result_id":"incomplete", "response":"\xe2\x82'
        with result_path.open("ab") as output:
            output.write(tail)
        with self.assertRaises(StorageError):
            read_results(path)
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertTrue(result_path.read_bytes().startswith(valid_prefix))
        self.assertEqual(len(read_results(path)), 5)
        backups = list(path.glob("results.jsonl.corrupt-*.bin"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), tail)

    def test_middle_corruption_and_conflicting_duplicates_fail_without_mutation(self):
        path = self.complete()
        result_path = path / "results.jsonl"
        original = result_path.read_bytes()
        result_path.write_bytes(b"broken\n" + original)
        before = result_path.read_bytes()
        with self.assertRaises(StorageError):
            execute_run(path, self.backend)
        self.assertEqual(result_path.read_bytes(), before)
        result_path.write_bytes(original)
        result = read_results(path)[0]
        append_jsonl(result_path, result)
        self.assertEqual(len(read_results(path)), 5)
        self.assertEqual(execute_run(path, self.backend), "complete")
        append_jsonl(result_path, {**result, "response": "conflicting response"})
        with self.assertRaisesRegex(RunError, "Widersprüchliche"):
            execute_run(path, self.backend)

    def test_complete_last_line_without_newline_stays_valid(self):
        path = self.create()
        self.backend.fail_at = 1
        execute_run(path, self.backend)
        result_path = path / "results.jsonl"
        original = result_path.read_bytes().rstrip(b"\n")
        result_path.write_bytes(original)
        self.assertEqual(execute_run(path, self.backend), "complete")
        self.assertTrue(result_path.read_bytes().startswith(original + b"\n"))
        self.assertEqual(len(read_results(path)), 5)
        self.assertEqual(list(path.glob("results.jsonl.corrupt-*")), [])

    def test_corrupt_metadata_reports_clear_error(self):
        path = self.create()
        (path / "metadata.json").write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(StorageError, "JSON kann nicht gelesen"):
            execute_run(path, self.backend)
        self.assertEqual(self.backend.calls, [])

    def test_human_reviews_append_history_and_never_modify_raw_answers(self):
        path = self.complete()
        raw = (path / "results.jsonl").read_bytes()
        baseline, steering = read_results(path)[:2]
        self.assertEqual(latest_evaluations(path), {})
        first = evaluate(path, steering["result_id"], "quality", 2)
        second = evaluate(path, steering["result_id"], "quality", 3)
        self.assertNotEqual(first["evaluation_id"], second["evaluation_id"])
        self.assertEqual(latest_evaluations(path)[(steering["result_id"], "quality")]["score"], 3)
        self.assertEqual(summarize(path)["review_counts"]["quality"], {"reviewed": 1, "total": 5})
        evaluate(path, steering["result_id"], "quality", None)
        self.assertIsNone(latest_evaluations(path)[(steering["result_id"], "quality")]["score"])
        self.assertEqual(summarize(path)["review_counts"]["quality"]["reviewed"], 0)
        evaluate(path, steering["result_id"], "steering_effect", -3)
        evaluate(path, baseline["result_id"], "refusal", 0)
        self.assertEqual(summarize(path)["review_counts"]["steering_effect"], {"reviewed": 1, "total": 4})
        self.assertEqual(len(read_jsonl(path / "evaluations.jsonl")), 5)
        self.assertEqual((path / "results.jsonl").read_bytes(), raw)
        self.assertEqual(first["reviewer_type"], "human")
        self.assertEqual(first["rubric_version"], "1")

    def test_review_validation_and_torn_evaluation_tail(self):
        path = self.complete()
        baseline, steering = read_results(path)[:2]
        for result_id, metric, score in (("unknown", "quality", 2), (baseline["result_id"], "steering_effect", 0),
                                         (steering["result_id"], "quality", 4), (steering["result_id"], "quality", True),
                                         (steering["result_id"], "quality", 1.0), (steering["result_id"], "unknown", 1)):
            with self.subTest(metric=metric, score=score):
                with self.assertRaises(ValueError):
                    evaluate(path, result_id, metric, score)
        self.assertEqual(latest_evaluations(path), {})
        with (path / "evaluations.jsonl").open("ab") as output:
            output.write(b'{"score":')
        evaluate(path, steering["result_id"], "quality", 2)
        self.assertEqual(latest_evaluations(path)[(steering["result_id"], "quality")]["score"], 2)
        self.assertEqual(len(list(path.glob("evaluations.jsonl.corrupt-*.bin"))), 1)

    def test_metric_rubrics_have_all_required_directions(self):
        self.assertEqual(set(METRICS), {"refusal", "quality", "steering_effect", "safety_risk"})
        self.assertEqual(METRICS["steering_effect"]["range"], list(range(-3, 4)))
        self.assertIn("höheres Risiko", METRICS["safety_risk"]["description"])


if __name__ == "__main__":
    unittest.main()
