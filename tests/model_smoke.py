"""Explicit harmless real-model verification; never runs in unittest discovery.

Run: .venv/bin/python tests/model_smoke.py
Creates seven short raw generations and a technical verification artifact.
Does not perform human evaluation or a full experiment.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rich.console import Console

from activation_steering.backend import Backend
from activation_steering.config import load_config
from activation_steering.datasets import load_dataset
from activation_steering.evaluation import summarize
from activation_steering.runs import ExperimentLock, create_run, discover_runs, execute_run, read_results
from activation_steering.storage import atomic_json
from activation_steering.vectors import build_vectors


def main():
    config = load_config("research")
    config.update(prompt_set="", prompt="Say hello.", steering_dataset="friendly_demo",
                  layers=[1, 32, 64], alphas=[-1.0, 0.0, 1.0],
                  temperature=0.0, seed=42, max_output_tokens=8)
    dataset = load_dataset("friendly_demo")
    with ExperimentLock("technical-smoke") as lock, Backend(config) as backend:
        steering = load_config("steering")
        vectors = build_vectors(backend, dataset, steering)
        assert len(vectors.vectors) == backend.layer_count == 64
        assert backend.dimension == 5120
        if steering["normalization"] == "unit_l2":
            assert all(abs(float(np.linalg.norm(v)) - 1.) < 1e-5 for v in vectors.vectors.values())
        else:
            assert all(
                abs(float(np.linalg.norm(v)) - float(vectors.metadata["raw_norms"][str(layer)]))
                <= 1e-5 * max(1.0, float(vectors.metadata["raw_norms"][str(layer)]))
                for layer, v in vectors.vectors.items()
            )
        try:
            with Backend(config):
                raise AssertionError("A second model loaded concurrently")
        except RuntimeError as error:
            assert "anderer CHAPPiE-Prozess" in str(error)
        run_dir = create_run(config, dataset, backend, vectors)
        print(f"SMOKE_RUN {run_dir}", flush=True)

        # Deliver actual SIGINT while native prompt evaluation is executing,
        # rather than only raising KeyboardInterrupt at a progress checkpoint.
        interruption = threading.Timer(.1, lambda: os.kill(os.getpid(), signal.SIGINT))
        armed = False
        def interrupt_during_generation(progress):
            nonlocal armed
            if not armed:
                armed = True
                interruption.start()
        try:
            assert execute_run(run_dir, backend, interrupt_during_generation, lock=lock) == "interrupted"
            assert not read_results(run_dir)
        finally:
            interruption.cancel()
            if armed:
                interruption.join()
        print("REAL_SIGINT_GENERATION interrupted cleanly", flush=True)

        def interrupt_after_baseline(progress):
            print(f"RUN {progress['completed']}/{progress['total']}", flush=True)
            if progress["completed"] == 1:
                assert len(read_results(run_dir)) == 1
                raise KeyboardInterrupt

        assert execute_run(run_dir, backend, interrupt_after_baseline, lock=lock) == "interrupted"
        baseline_bytes = (run_dir / "results.jsonl").read_bytes()
        assert execute_run(run_dir, backend, lambda p: print(f"RESUME {p['completed']}/{p['total']}", flush=True), lock=lock) == "complete"
        raw = (run_dir / "results.jsonl").read_bytes()
        assert raw.startswith(baseline_bytes)
        results = read_results(run_dir)
        assert len(results) == 7 and sum(r["type"] == "baseline" for r in results) == 1
        assert len({r["result_id"] for r in results}) == 7
        assert (run_dir / "evaluations.jsonl").read_bytes() == b""
        assert run_dir in discover_runs()
        control = {}
        for layer in (1, 32, 64):
            probe = backend.alpha_zero_control(config, vectors.vectors[layer], layer)
            assert probe["passed"], f"Alpha-0 mismatch in layer {layer}"
            control[str(layer)] = probe
            print(f"ALPHA_ZERO {layer} passed", flush=True)

        # Verify an actual numerical change in model logits, independently of
        # whether a small steering strength changes the greedy output tokens.
        prompt = [{"role": "user", "content": config["prompt"]}]
        one_token = dict(config, max_output_tokens=1)
        def logits(vector=None, layer=None, alpha=0):
            backend.generate(prompt, one_token, vector=vector, layer=layer, alpha=alpha)
            return np.ctypeslib.as_array(backend.model._ctx.get_logits(), shape=(backend.model.n_vocab(),)).copy()
        baseline_logits = logits()
        differences = {}
        for layer in (1, 32, 64):
            shifted = logits(vectors.vectors[layer], layer, 1.0)
            difference = float(np.max(np.abs(shifted - baseline_logits)))
            assert np.isfinite(difference) and difference > 0
            differences[str(layer)] = difference
            assert np.array_equal(logits(), baseline_logits), f"Steering state leaked after layer {layer}"
            print(f"LOGIT_INJECTION {layer} changed and reset", flush=True)

        # Exercise the visible chat flow against the same real model, retaining
        # the OS model lock and avoiding any second weight load. Config changes
        # are test-local, with /new selecting the next complete conversation.
        from cli import chat, common, matrix, review
        from cli.research import preview_run
        transcript = io.StringIO()
        terminal = Console(file=transcript, width=80, height=32, color_system=None)
        plain = load_config("chat")
        plain.update(temperature=0.0, seed=42, max_output_tokens=8)
        steered = deepcopy(plain)
        steered["steering"] = {"enabled": True, "dataset": dataset.name, "layer": 32, "alpha": 1.0}
        def configs(kind):
            if kind == "chat":
                return deepcopy(next(chat_configs))
            return load_config(kind)
        @contextmanager
        def reuse_backend(_config):
            yield backend
            backend.reset()
        chat_configs = iter([plain, steered])
        with patch.object(common, "console", terminal), patch.object(chat, "console", terminal), \
             patch.object(matrix, "console", terminal), patch.object(chat, "load_config", side_effect=configs), \
             patch("activation_steering.backend.Backend", side_effect=reuse_backend), \
             patch("builtins.input", side_effect=["/status", "Say hello.", "/new", "/status", "Say hello.", "/exit"]):
            chat.chat_session()
        with patch.object(common, "console", terminal), patch.object(matrix, "console", terminal):
            preview_run(config, dataset, backend, load_config("steering"))
            review.show_result(run_dir, results[-1])
            for metric in matrix.MATRIX_METRICS:
                matrix.render_matrix(run_dir, metric)
            with patch("builtins.input", side_effect=["6"]):
                review.review_inbox(run_dir)
        rendered = transcript.getvalue()
        for expected in ("Neuer Chat", "Steering:", "Prompt Tokens", "TTFT", "TPS", "Gemeinsame Baseline", "Review Inbox"):
            assert expected in rendered, expected
        assert hashlib.sha256((run_dir / "results.jsonl").read_bytes()).digest() == hashlib.sha256(raw).digest()
        assert (run_dir / "evaluations.jsonl").read_bytes() == b""
        (run_dir / "terminal_verification.txt").write_text(rendered, encoding="utf-8")
        evidence = {"technical_smoke_only": True, "research_use": False, "status": "passed",
            "run_dir": run_dir.name, "decoder_layers": backend.layer_count, "vector_dimension": backend.dimension,
            "raw_generations": 7, "interrupt_resume": True, "real_sigint_during_generation": True, "single_model_lock": True,
            "alpha_zero_controls": control, "nonzero_logit_max_abs_differences": differences,
            "baseline_reset_logits_exact": True, "real_chat_baseline_and_steering": True,
            "history_review_matrix_terminal": True, "review_scores_fabricated": False,
            "summary": summarize(run_dir)}
        atomic_json(run_dir / "verification.json", evidence)
        print(json.dumps({"status": "passed", "run": str(run_dir), "generations": 7}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
