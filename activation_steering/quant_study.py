"""Prepare, run and compare the pinned Qwen3.8-27B quantization pilot."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import tomllib

from .config import CONFIG_DIR, ROOT, validate_config
from .datasets import load_dataset
from .prompts import load_prompt_set
from .runs import ExperimentLock, create_run, execute_run, plan_run, read_metadata, read_results


SETTINGS = CONFIG_DIR / "quantization_pilot.toml"


def load_pilot(path):
    raw = path.read_bytes()
    study = tomllib.loads(raw.decode("utf-8"))
    config = validate_config("research", study["research"])
    steering = validate_config("steering", study["steering"])
    if steering["normalization"] != "none" or steering["cache"]:
        raise ValueError("Der Pilot verlangt unnormierte und frisch extrahierte Vektoren.")
    prompts = load_prompt_set(config["prompt_set"])
    dataset = load_dataset(config["steering_dataset"])
    if prompts.hash != study["inputs"]["prompt_set_sha256"] or dataset.hash != study["inputs"]["dataset_sha256"]:
        raise ValueError("Prompt-Set oder Steering-Datensatz weicht vom eingefrorenen Pilot ab.")
    labels = {entry["quant"] for entry in study["variants"]}
    if len(labels) != len(study["variants"]) or study["source"]["reference"] not in labels:
        raise ValueError("Die Quantisierungsliste enthält doppelte Varianten oder keine Referenz.")
    if any(Path(entry["file"]).name != entry["file"] for entry in study["variants"]):
        raise ValueError("Varianten erwarten Dateinamen im Modellordner.")
    config.update(prompts=[item["prompt"] for item in prompts.prompts],
                  prompt_ids=[item["id"] for item in prompts.prompts])
    study["config_sha256"] = sha256(raw).hexdigest()
    return study, config, steering, dataset, prompts


def verify_model(path, variant):
    from .backend import cached_hash

    if path.stat().st_size != variant["bytes"] or cached_hash(path) != variant["sha256"]:
        raise ValueError(f"{path}: Größe oder SHA-256 passt nicht zur gepinnten GGUF-Datei.")


def verify_backend(backend, study, variant):
    expected = study["inputs"]
    if backend.metadata["model_hash"] != variant["sha256"]:
        raise ValueError("Die Modelldatei hat sich zwischen Prüfung und Laden geändert.")
    if backend.metadata["quantization"] != variant["quant"]:
        raise ValueError(f"GGUF-Dateityp passt nicht zu {variant['quant']}: {backend.metadata['gguf_file_type']}.")
    if (backend.layer_count, backend.dimension) != (expected["decoder_layers"], expected["dimension"]):
        raise ValueError("Decoder-Layerzahl oder Vektordimension weicht vom Pilot ab.")


def smoke(backend, config, steering, inputs):
    import numpy as np

    from .vectors import build_vectors

    # build_vectors calls real l_out extraction and validates all decoder layers.
    vectors = build_vectors(backend, load_dataset("friendly_demo"), steering)
    sample = dict(config, prompt=inputs["smoke_prompt"], temperature=0.0, max_output_tokens=8)
    probe = dict(sample, max_output_tokens=1)
    messages = [{"role": "user", "content": sample["prompt"]}]

    def logits(vector=None, layer=None, alpha=0.0):
        backend.generate(messages, probe, vector, layer, alpha)
        values = np.ctypeslib.as_array(backend.model._ctx.get_logits(), shape=(backend.model.n_vocab(),)).copy()
        if not np.isfinite(values).all():
            raise ValueError("Smoke-Test: nicht-endliche Logits.")
        return values

    baseline = logits()
    checks = {}
    for layer in inputs["smoke_layers"]:
        vector = vectors.vectors[layer]
        check = backend.alpha_zero_control(sample, vector, layer)
        shifted = logits(vector, layer, 1.0)
        maximum = float(np.max(np.abs(shifted.astype(np.float64) - baseline)))
        if not check["passed"] or not (maximum > 0) or not np.array_equal(logits(), baseline):
            raise ValueError(f"Smoke-Test fehlgeschlagen in Layer {layer}: Alpha 0, Logits oder Reset.")
        checks[str(layer)] = {"alpha_zero": check, "max_abs_logit_delta": maximum}
    return checks


def tokenizer_fingerprint(backend, dataset, config):
    tokenizer = backend.model.tokenize
    train = [tokenizer(pair[key].encode("utf-8"), add_bos=True, special=False)
             for pair in dataset.pairs for key in ("target", "contrast")]
    evaluation = [backend.tokens(backend.format_prompt([{"role": "user", "content": prompt}], config["enable_thinking"]))
                  for prompt in config["prompts"]]
    return {
        "model": backend.model.metadata.get("tokenizer.ggml.model"),
        "pre": backend.model.metadata.get("tokenizer.ggml.pre"),
        "vocab_size": backend.model.n_vocab(),
        "study_token_ids_sha256": sha256(json.dumps([train, evaluation]).encode("utf-8")).hexdigest(),
    }


def run_quant(variant, args, study, config, steering, dataset, prompts):
    from .backend import Backend, _command
    from .vectors import build_vectors

    path = ROOT / "models" / variant["file"]
    config = dict(config, model_path=str(path), n_gpu_layers=args.gpu_layers)
    if args.seed is not None:
        config["seed"] = args.seed
    # Runtime-only prompt arrays are added after validation of the research schema.
    validate_config("research", {key: value for key, value in config.items() if key not in ("prompts", "prompt_ids")})
    verify_model(path, variant)

    gpu_before = _command(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,driver_version", "--format=csv"])
    with ExperimentLock("quantization-pilot") as lock, Backend(config, verbose=True) as backend:
        verify_backend(backend, study, variant)
        checks = smoke(backend, config, steering, study["inputs"])
        report = {
            "source": study["source"], "variant": variant, "profile": args.profile,
            "config_sha256": study["config_sha256"], "gpu_before_load": gpu_before,
            "model_load_seconds": backend.load_time, "smoke": checks,
            "tokenizer": tokenizer_fingerprint(backend, dataset, config),
            "offload_argument": backend.metadata["n_gpu_layers"],
            "cpu_threads": backend.metadata["n_threads"],
        }
        if args.smoke_only:
            return {"status": "smoke_passed", **report, "backend": backend.metadata}
        if args.profile == "performance":
            report["baseline_prefill_seconds"] = {
                prompt_id: backend.measure_prefill([{"role": "user", "content": prompt}], config)
                for prompt_id, prompt in zip(config["prompt_ids"], config["prompts"], strict=True)
            }
        started = perf_counter()
        vectors = build_vectors(backend, dataset, steering)
        report["vector_build_seconds"] = perf_counter() - started
        backend.metadata["quantization_pilot"] = report
        run_dir = create_run(config, dataset, backend, vectors, prompt_set=prompts)
        print(f"Run: {run_dir}", file=sys.stderr, flush=True)
        status = execute_run(run_dir, backend, lock=lock)
        return {"status": status, "run_dir": str(run_dir), **report}


def compare(reference, other):
    import numpy as np

    from .evaluation import METRICS, RUBRIC_VERSION, latest_evaluations
    from .vectors import load_vectors

    left, right = (read_metadata(path) for path in (reference, other))
    a, b = (meta.get("quantization_pilot") for meta in (left, right))
    if not a or not b or left["status"] != "complete" or right["status"] != "complete":
        raise ValueError("Vergleich erwartet zwei vollständige Quantisierungspiloten.")
    if a["variant"]["quant"] != a["source"]["reference"] or a["variant"] == b["variant"]:
        raise ValueError("Zuerst den IQ4_XS-Referenzrun, dann einen anderen Quantisierungsrun angeben.")
    controls = ("dataset_hash", "prompt_set_hash", "layers", "alphas", "generation_settings", "architecture",
                "layer_count", "dimension", "context_size", "chat_template_hash", "llama_cpp_python_version",
                "backend_library_hashes", "bridge_version", "cpu", "gpu", "n_threads")
    study_controls = ("source", "config_sha256", "profile", "tokenizer")
    if any(left.get(key) != right.get(key) for key in controls) or any(a[key] != b[key] for key in study_controls):
        raise ValueError("Studienbedingungen unterscheiden sich.")
    if a["profile"] == "quality" and left["n_gpu_layers"] != right["n_gpu_layers"]:
        raise ValueError("Qualitätsvergleich erwartet dieselbe Offload-Einstellung.")

    vector_path = "vectors.npz"
    for path, metadata in ((reference, left), (other, right)):
        if sha256((path / vector_path).read_bytes()).hexdigest() != metadata["artifact_hashes"].get(vector_path):
            raise ValueError(f"{path}: Vektorartefakt stimmt nicht mit dem Run-Snapshot überein.")
    v_ref, v_other = (load_vectors(path / vector_path) for path in (reference, other))
    if v_ref.vectors.keys() != v_other.vectors.keys():
        raise ValueError("Die Vektorsätze enthalten verschiedene Layer.")
    vectors = {}
    for layer in v_ref.vectors:
        x, y = (item.vectors[layer].astype(np.float64) for item in (v_ref, v_other))
        length = float(np.linalg.norm(x) * np.linalg.norm(y))
        vectors[str(layer)] = {
            "reference_l2": v_ref.metadata["raw_norms"][str(layer)],
            "other_l2": v_other.metadata["raw_norms"][str(layer)],
            "cosine": float(np.clip(np.dot(x, y) / length, -1, 1)) if length else None,
        }

    left_results, right_results = (read_results(path) for path in (reference, other))
    identities = ("result_id", "prompt_id", "prompt", "type", "layer", "alpha")
    if len(left_results) != left["total"] or len(right_results) != right["total"]:
        raise ValueError("Vollständige Rohantworten fehlen.")
    right_results = {item["result_id"]: item for item in right_results}
    ratings = [latest_evaluations(path) for path in (reference, other)]
    baselines = {item["prompt_id"]: item["result_id"] for item in left_results if item["type"] == "baseline"}

    def score(collection, result_id, metric):
        review = collection.get((result_id, metric), {})
        return review.get("score") if review.get("rubric_version") == RUBRIC_VERSION else None

    rows = []
    for first in left_results:
        second = right_results.get(first["result_id"])
        if second is None or any(first[key] != second[key] for key in identities):
            raise ValueError("Die Rohantworten lassen sich nicht nach derselben Prompt-/Layer-/Alpha-Zelle paaren.")
        row = {key: first[key] for key in identities if key != "prompt"}
        row["runtime"] = [
            {key: item.get(key) for key in ("ttft", "tokens_per_second", "output_tokens", "finish_reason")}
            for item in (first, second)
        ]
        row["reviews"] = {}
        for metric in METRICS:
            pairs = [[score(collection, first["result_id"], metric),
                      score(collection, baselines[first["prompt_id"]], metric)] for collection in ratings]
            row["reviews"][metric] = [{"score": value, "baseline": baseline,
                                       "delta": value - baseline if value is not None and baseline is not None else None}
                                      for value, baseline in pairs]
        rows.append(row)
    return {"reference": left["run_id"], "other": right["run_id"], "variants": [a["variant"], b["variant"]],
            "profile": a["profile"], "pilot_reports": [a, b], "vectors": vectors, "paired_responses": rows}


def gpu_layers(value):
    if value == "auto":
        return value
    try:
        number = int(value)
        if number >= -1:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("erwartet auto, -1, 0 oder eine positive Layerzahl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="Matrix und HF-Dateifingerprints ohne Modell anzeigen")
    run = commands.add_parser("run", help="Eine lokale Quantisierung prüfen und optional den Pilot ausführen")
    run.add_argument("quant")
    run.add_argument("--gpu-layers", required=True, type=gpu_layers)
    run.add_argument("--profile", choices=("quality", "performance"), default="quality")
    run.add_argument("--seed", type=int)
    run.add_argument("--smoke-only", action="store_true")
    report = commands.add_parser("compare", help="IQ4_XS-Referenz und einen vollständigen Run paaren")
    report.add_argument("reference", type=Path)
    report.add_argument("other", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            output = compare(args.reference, args.other)
        else:
            study, config, steering, dataset, prompts = load_pilot(args.settings)
            if args.command == "plan":
                plan = plan_run(config, study["inputs"]["decoder_layers"])
                output = {key: study[key] for key in ("source", "inputs", "variants", "excluded", "config_sha256")}
                output["plan"] = {key: plan[key] for key in ("layers", "alphas", "prompt_ids", "baseline", "steering", "total")}
            else:
                variants = {item["quant"]: item for item in study["variants"]}
                if args.quant not in variants:
                    parser.error(f"Unbekannte Quantisierung: {args.quant}. Erlaubt: {', '.join(variants)}.")
                if args.profile == "quality" and args.gpu_layers in ("auto", -1):
                    parser.error("Für quality eine feste gemeinsame Layerzahl ab 0 angeben.")
                output = run_quant(variants[args.quant], args, study, config, steering, dataset, prompts)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 130 if output.get("status") == "interrupted" else 0
    except KeyboardInterrupt:
        print("Pilot unterbrochen.", file=sys.stderr)
        return 130
    except (KeyError, OSError, ValueError, RuntimeError) as error:
        print(f"Quantisierungspilot: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
