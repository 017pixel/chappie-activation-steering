"""Exercise frozen selections and analysis with fixture artifacts only."""

from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest

import numpy as np

from activation_steering.backend import BRIDGE_VERSION
from activation_steering.config import load_config
from activation_steering.datasets import load_dataset
from activation_steering.evaluation import evaluate
from activation_steering.prompts import SHARED_SYCOPHANCY, load_prompt_set
from activation_steering.runs import create_run, execute_run, read_metadata, read_results
from activation_steering.stability import FRAMES, SEEDS, SIZES, THEME_SIZE, TOPICS, check_leakage, prepare
from activation_steering.stability_analysis import analyze, cosine_row
from activation_steering.storage import atomic_json, read_json
from activation_steering.vectors import build_vectors
from test_runs_evaluation import FakeBackend


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.study = cls.directory / "selection"
        prepare(cls.study)
        cls.manifest = read_json(cls.study / "manifest.json")
        cls.holdout = load_prompt_set(cls.manifest["prompt_set_name"], cls.study)

    def test_materialization_is_stable_and_loadable(self):
        repeat = self.directory / "repeat"
        subprocess.run([sys.executable, "-m", "activation_steering.stability", repeat],
                       check=True, capture_output=True, text=True)
        self.assertEqual({path.name: path.read_bytes() for path in self.study.iterdir()},
                         {path.name: path.read_bytes() for path in repeat.iterdir()})
        with self.assertRaises(FileExistsError):
            prepare(repeat)

        conditions = self.manifest["conditions"]
        self.assertEqual(len(conditions), 31)
        self.assertEqual(self.holdout.hash, self.manifest["prompt_set_hash"])
        shared = load_prompt_set(SHARED_SYCOPHANCY)
        self.assertEqual(self.holdout.source, shared.source)
        self.assertEqual(self.manifest["prompt_ids"], [row["id"] for row in shared.prompts])
        self.assertEqual(self.manifest["responses_per_run"], 100)
        self.assertEqual(self.manifest["planned_responses"], 3100)
        self.assertEqual(len({tuple(item["claim_ids"]) for item in conditions}), len(conditions))
        reference = next(item for item in conditions if item["group"] == "reference")
        self.assertEqual(reference["size"], 48)
        for item in conditions:
            dataset = load_dataset(item["name"], self.study)
            self.assertEqual(dataset.hash, item["dataset_hash"])
            self.assertEqual(len(dataset.pairs), item["size"])
            rows = tomllib.loads(dataset.source)["pairs"]
            self.assertEqual([row["id"] for row in rows], item["claim_ids"])
            self.assertTrue(set(item["claim_ids"]) <= set(reference["claim_ids"]))
            self.assertTrue(set(item["claim_ids"]).isdisjoint(self.manifest["prompt_ids"]))
            if item["group"] in TOPICS:
                self.assertEqual(item["size"], THEME_SIZE)
                self.assertEqual(Counter(row["topic"] for row in rows), {item["group"]: THEME_SIZE})
                self.assertEqual(Counter(row["frame"] for row in rows),
                                 {frame: THEME_SIZE // len(FRAMES) for frame in range(1, len(FRAMES) + 1)})
        for size in SIZES:
            self.assertEqual([item["seed"] for item in conditions if item["group"] == "size" and item["size"] == size],
                             list(SEEDS))
        # Freeze one independent draw as a regression guard for the protocol.
        self.assertEqual(conditions[0]["claim_ids"],
                         ["impressionism-name", "machu-picchu-culture", "nitrogen-protons",
                          "nobel-founder", "periscope-mirrors"])

    def test_leakage_checks_ids_exact_text_and_small_edits(self):
        claims = [("stickstoff-protonen", ["Ein Stickstoffatom hat acht Protonen.",
                                           "Ein Stickstoffatom hat sieben Protonen."])]
        for test in (
            [("stickstoff-protonen", ["Andere Wörter."])],
            [("andere-id", ["EIN STICKSTOFFATOM HAT SIEBEN PROTONEN!"])],
            [("andere-id", ["Ein Stickstoffatom hat ungefähr sieben Protonen."])],
        ):
            with self.subTest(test=test), self.assertRaises(ValueError):
                check_leakage(claims, test)
        check_leakage(claims, [("teppich", ["Der Teppich von Bayeux ist gewebt."])])

    def fixture_run(self, directory, name, direction):
        manifest = self.manifest
        config = load_config("research")
        config.update(manifest["generation"], prompt_set=self.holdout.name, model_path=manifest["model_file"],
                      context_size=manifest["context_size"], layers=manifest["layers"], alphas=manifest["alphas"],
                      steering_dataset=name, prompts=[prompt["prompt"] for prompt in self.holdout.prompts],
                      prompt_ids=manifest["prompt_ids"])
        dataset = load_dataset(name, self.study)
        backend = FakeBackend(config, layers=max(manifest["layers"]))
        backend.metadata.update(model_file=config["model_path"], quantization=manifest["quantization"],
                                bridge_version=BRIDGE_VERSION)
        targets = {pair["target"] for pair in dataset.pairs}
        backend.extract = lambda text: {
            layer: np.array(direction if text in targets else [0, 0, 0], np.float32)
            for layer in range(1, backend.layer_count + 1)
        }
        steering = {"cache": False, "normalization": manifest["normalization"],
                    "token_position": manifest["token_position"]}
        vectors = build_vectors(backend, dataset, steering)
        run = create_run(config, dataset, backend, vectors, directory, self.holdout)
        # Leave an interrupted run containing the first two prompts for coverage checks.
        backend.fail_at = 20
        self.assertEqual(execute_run(run, backend), "interrupted")
        return run

    def review_pair(self, path, prompt_id, base_score, steered_score):
        answers = [answer for answer in read_results(path)
                   if answer["prompt_id"] == prompt_id and answer["layer"] in (None, 8) and answer["alpha"] in (0, -1)]
        baseline, answer = answers
        evaluate(path, baseline["result_id"], "sycophancy", base_score)
        evaluate(path, answer["result_id"], "sycophancy", steered_score)
        return answer["result_id"]

    def test_analysis_reads_run_snapshots_and_pairs_reviews(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as folder:
            directory = Path(folder)
            a = self.fixture_run(directory, "stability_size_5_s17", [3, 4, 0])
            b = self.fixture_run(directory, "stability_size_5_s43", [-3, -4, 0])
            first, second = self.manifest["prompt_ids"][:2]
            answer_id = self.review_pair(a, first, 2, 0)
            self.review_pair(a, second, 1, 3)
            self.review_pair(b, first, 3, 2)

            report = analyze(self.manifest, [b, a])
            self.assertEqual(report, analyze(self.manifest, [a, b]))
            group = report["groups"][0]
            self.assertEqual((group["available_runs"], group["planned_runs"]), (2, 5))
            self.assertEqual(len(report["missing_datasets"]), 29)
            self.assertEqual(group["geometry"][0]["l2"]["mean"], 5)
            self.assertEqual(group["geometry"][0]["cosine"]["mean"], -1)
            self.assertEqual(group["geometry"][0]["cosine"]["n"], 1)
            self.assertIsNone(group["geometry"][0]["cosine"]["sd"])
            metrics = group["effects"][0]["metrics"]
            self.assertEqual(metrics["sycophancy"]["prompts"], [first])
            self.assertEqual(metrics["sycophancy"]["paired_reviews_by_run"], {a.name: 2, b.name: 1})
            self.assertEqual(metrics["sycophancy"]["delta"]["mean"], -1.5)
            self.assertIsNone(metrics["quality"]["delta"]["mean"])
            self.assertEqual(report["runs"][0]["conditions"][2]["intervention_l2"], 10)
            self.assertEqual(report["runs"][0]["generated"], 20)
            self.assertEqual(report["runs"][0]["status"], "interrupted")
            self.assertEqual(report["baseline_differences"], [])
            output = directory / "report.json"
            command = [sys.executable, "-m", "activation_steering.stability_analysis",
                       self.study / "manifest.json", a, b, "--output", output]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(read_json(output)["groups"], report["groups"])
            original = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), original)

            evaluate(a, answer_id, "sycophancy", None)
            metrics = analyze(self.manifest, [a, b])["groups"][0]["effects"][0]["metrics"]
            self.assertEqual(metrics["sycophancy"]["prompts"], [])
            self.assertEqual(metrics["sycophancy"]["delta"]["n"], 0)

            metadata = read_metadata(b)
            metadata["backend_fingerprint"]["model_hash"] = "other model"
            atomic_json(b / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "Modell, Backend"):
                analyze(self.manifest, [a, b])
            with (a / "prompt_set.toml").open("ab") as file:
                file.write(b"# changed\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                analyze(self.manifest, [a])

    def test_theme_comparison_pairs_same_seed_and_same_prompts(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as folder:
            directory = Path(folder)
            a = self.fixture_run(directory, "stability_naturwissenschaft_s17", [1, 0, 0])
            b = self.fixture_run(directory, "stability_geschichte_kultur_s17", [0, 1, 0])
            first, second = self.manifest["prompt_ids"][:2]
            self.review_pair(a, first, 2, 0)
            self.review_pair(b, first, 3, 2)
            self.review_pair(b, second, 1, 3)
            comparison = analyze(self.manifest, [b, a])["topic_comparison"]
            self.assertEqual(comparison["seeds"], [17])
            self.assertEqual(comparison["geometry"][0]["cosine"]["mean"], 0)
            self.assertEqual(comparison["geometry"][0]["pairs"][0]["shared_claims"], 0)
            metric = comparison["effects"][0]["metrics"]["sycophancy"]
            self.assertEqual(metric["prompts"], [first])
            self.assertEqual(metric["difference"]["mean"], -1)

    def test_cosine_with_zero_vector_is_undefined(self):
        a = SimpleNamespace(vectors={8: np.zeros(3)}, metadata={"run_id": "zero"},
                            condition={"claim_ids": ["a"]})
        self.assertIsNone(cosine_row(a, a, 8)["cosine"])
