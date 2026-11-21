"""Model-free checks for the separate quantization research entry point."""

from argparse import Namespace
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import numpy as np

from activation_steering import quant_study
from activation_steering.backend import quantization_name
from activation_steering.prompts import SHARED_SYCOPHANCY, load_prompt_set
from activation_steering.runs import read_metadata, read_snapshot_config
from activation_steering.storage import atomic_json
from test_runs_evaluation import FakeBackend


class QuantStudyTests(unittest.TestCase):
    def test_plan_uses_all_pinned_variants_and_one_baseline_per_prompt(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(quant_study.main(["plan"]), 0)
        planned = json.loads(output.getvalue())
        plan = planned["plan"]
        shared = load_prompt_set(SHARED_SYCOPHANCY)
        self.assertEqual(planned["inputs"]["prompt_set_sha256"], shared.hash)
        self.assertEqual(plan["prompt_ids"], [row["id"] for row in shared.prompts])
        self.assertEqual((plan["baseline"], plan["steering"], plan["total"]), (10, 90, 100))
        self.assertEqual(plan["total_per_profile"], 1200)
        variants = {item["quant"]: item for item in planned["variants"]}
        self.assertEqual(set(variants), {
            "IQ4_XS", "IQ3_M", "Q3_K_L", "Q3_K_M", "IQ3_XS", "Q3_K_S",
            "IQ3_XXS", "Q2_K", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
        })
        for quant, item in variants.items():
            self.assertEqual(item["file"], f"Qwen3.8-27B-{quant}.gguf")
            self.assertRegex(item["sha256"], "^[0-9a-f]{64}$")
            if quant != "IQ4_XS":
                self.assertLess(item["bytes"], variants["IQ4_XS"]["bytes"])
        self.assertIn("Q2_K_L", planned["excluded"][0]["file"])

    def test_gguf_type_wins_over_file_name_and_known_name_handles_missing_type(self):
        labels = {
            10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 19: "IQ2_XXS",
            20: "IQ2_XS", 22: "IQ3_XS", 23: "IQ3_XXS", 27: "IQ3_M",
            28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS",
        }
        llama = SimpleNamespace(LLAMA_FTYPE_ALL_F32=0, **{
            f"LLAMA_FTYPE_MOSTLY_{label}": code for code, label in labels.items()
        })
        for code, expected in labels.items():
            with self.subTest(quant=expected):
                self.assertEqual(quantization_name(llama, str(code), "unrelated.gguf"), expected)
                self.assertEqual(quantization_name(llama, None, f"Qwen3.8-27B-{expected}.gguf"), expected)
        self.assertEqual(quantization_name(llama, 12, "old-IQ4_XS.gguf"), "Q3_K_M")
        self.assertIsNone(quantization_name(llama, None, "unknown.gguf"))

    def test_file_fingerprint_must_match(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.gguf"
            path.write_bytes(b"fixture")
            with patch("activation_steering.backend.cached_hash", return_value="actual"):
                quant_study.verify_model(path, {"bytes": 7, "sha256": "actual"})
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    quant_study.verify_model(path, {"bytes": 7, "sha256": "wrong"})

    def test_separate_snapshots_and_model_free_comparison(self):
        pilot = quant_study.load_pilot(quant_study.SETTINGS)
        study = pilot[0]
        study["inputs"]["dimension"] = 3
        variants = {item["file"]: item for item in study["variants"]}
        args = Namespace(profile="quality", gpu_layers=0, seed=None, smoke_only=False)

        class StudyBackend(FakeBackend):
            def __init__(self, config, *, verbose):
                super().__init__(config, layers=64)
                variant = variants[Path(config["model_path"]).name]
                self.metadata.update(model_hash=variant["sha256"], quantization=variant["quant"],
                                     chat_template_hash=study["inputs"]["chat_template_sha256"])
                self.load_time = 0.1

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def extract(self, sample):
                checksum = sum(sample.encode("utf-8"))
                return {layer: np.array([checksum, layer, 1], dtype=np.float32) for layer in range(1, 65)}

        with TemporaryDirectory() as directory, patch("activation_steering.runs.RUNS_DIR", Path(directory)), \
                patch("activation_steering.backend.Backend", StudyBackend), \
                patch("activation_steering.backend._command", return_value=None), \
                patch.object(quant_study, "verify_model"), patch.object(quant_study, "smoke", return_value={}), \
                patch.object(quant_study, "tokenizer_fingerprint", return_value={"fixture": True}), \
                redirect_stderr(io.StringIO()):
            first, second = (quant_study.run_quant(variant, args, *pilot) for variant in study["variants"][:2])
            self.assertEqual(first["status"], "complete")
            self.assertEqual(second["status"], "complete")
            reference, other = Path(first["run_dir"]), Path(second["run_dir"])
            self.assertNotEqual(reference, other)
            shared = load_prompt_set(SHARED_SYCOPHANCY)
            for path in (reference, other):
                self.assertEqual((path / "prompt_set.toml").read_bytes(), shared.source.encode("utf-8"))
            self.assertEqual(read_snapshot_config(reference)["seed"], 42)
            self.assertNotEqual(read_metadata(reference)["model_hash"], read_metadata(other)["model_hash"])
            result = quant_study.compare(reference, other)
            self.assertEqual(len(result["paired_responses"]), 100)
            self.assertEqual(len(result["vectors"]), 64)
            self.assertAlmostEqual(result["vectors"]["1"]["cosine"], 1.0)
            self.assertIsNone(result["paired_responses"][0]["reviews"]["quality"][0]["delta"])

            metadata = read_metadata(other)
            metadata["n_gpu_layers"] = 1
            atomic_json(other / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "Offload"):
                quant_study.compare(reference, other)
