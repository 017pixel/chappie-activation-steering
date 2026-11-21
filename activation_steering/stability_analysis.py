"""Compare vector and human-review snapshots for the fixed stability study."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations, product
import math
from pathlib import Path
from statistics import mean, stdev
import tomllib

import numpy as np

from .datasets import load_dataset
from .evaluation import METRICS, RUBRIC_VERSION, latest_evaluations
from .prompts import load_prompt_set
from .runs import read_metadata, read_results, read_snapshot_config
from .stability import PROTOCOL_VERSION, TOPICS
from .storage import json_bytes, read_json
from .vectors import load_vectors


@dataclass
class Sample:
    condition: dict
    metadata: dict
    vectors: dict
    norms: dict
    vector_hash: str
    comparisons: dict
    baseline_tokens: dict
    generated: int
    generation_seconds: float


def load_sample(path, metadata, condition, manifest):
    expected = {
        "steering_dataset.toml": condition["dataset_hash"],
        "prompt_set.toml": manifest["prompt_set_hash"],
        "config.toml": metadata["artifact_hashes"].get("config.toml"),
        "steering_metadata.json": metadata["artifact_hashes"].get("steering_metadata.json"),
        "vectors.npz": metadata["artifact_hashes"].get("vectors.npz"),
    }
    for name, digest in expected.items():
        if metadata["artifact_hashes"].get(name) != digest or hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{path}: SHA-256 von {name} weicht ab.")

    dataset = load_dataset(path / "steering_dataset.toml")
    prompt_set = load_prompt_set(path / "prompt_set.toml")
    prompt_texts = {prompt["id"]: prompt["prompt"] for prompt in prompt_set.prompts}
    config = read_snapshot_config(path)
    layers, alphas, prompt_ids = (manifest[key] for key in ("layers", "alphas", "prompt_ids"))
    if (not dataset.research_use or dataset.name != condition["name"]
            or metadata["dataset_hash"] != dataset.hash
            or [pair["id"] for pair in tomllib.loads(dataset.source)["pairs"]] != condition["claim_ids"]
            or prompt_set.name != manifest["prompt_set_name"] or list(prompt_texts) != prompt_ids
            or metadata.get("prompt_set_hash") != prompt_set.hash):
        raise ValueError(f"{path}: Dataset oder Holdout weicht vom Manifest ab.")
    if (Path(metadata["model_file"]).name != manifest["model_file"]
            or metadata["quantization"] != manifest["quantization"]
            or not metadata.get("model_hash")
            or config["resolved"] != {"layers": layers, "alphas": alphas, "prompt_ids": prompt_ids,
                                      "prompts": list(prompt_texts.values())}
            or config["context_size"] != manifest["context_size"]
            or any(config[key] != value for key, value in manifest["generation"].items())):
        raise ValueError(f"{path}: Modell oder Run-Einstellungen weichen vom Protokoll ab.")

    vector_set = load_vectors(path / "vectors.npz")
    extraction = vector_set.metadata["extraction_settings"]
    checks = {key: manifest[key] for key in ("normalization", "token_position", "context_size")}
    checks.update(dataset_hash=dataset.hash, model_hash=metadata["model_hash"], n_gpu_layers=config["n_gpu_layers"],
                  n_threads=config["n_threads"], backend_library_hashes=metadata["backend_library_hashes"],
                  backend_version=metadata["llama_cpp_python_version"], method_version=metadata["bridge_version"],
                  text_format="raw_text_add_bos_true_special_false_no_padding")
    if (any(extraction.get(key) != value for key, value in checks.items())
            or vector_set.metadata["pair_count"] != condition["size"]
            or not set(layers).issubset(vector_set.vectors)):
        raise ValueError(f"{path}: Vektorextraktion weicht vom Protokoll ab.")

    vectors = {layer: vector_set.vectors[layer].astype(np.float64) for layer in layers}
    norms = {layer: vector_set.metadata["raw_norms"][str(layer)] for layer in layers}
    for layer in layers:
        if not math.isclose(norms[layer], np.linalg.norm(vectors[layer]), rel_tol=1e-6):
            raise ValueError(f"{path}: L2-Norm und Vektor in Layer {layer} passen nicht zusammen.")

    results = read_results(path)
    latest = latest_evaluations(path)
    result_ids = {result["result_id"] for result in results}
    if any(result_id not in result_ids or entry["rubric_version"] != manifest["rubric_version"]
           for (result_id, _), entry in latest.items()):
        raise ValueError(f"{path}: Bewertung ohne passende Antwort oder mit abweichender Rubrik.")
    expected_keys = {(prompt_id, None, 0.0) for prompt_id in prompt_ids}
    expected_keys.update(product(prompt_ids, layers, alphas))
    records = {}
    for result in results:
        key = result["prompt_id"], result["layer"], result["alpha"]
        if (key not in expected_keys or key in records or result["prompt"] != prompt_texts[key[0]]
                or result["repeat_id"] != 0 or result["seed"] != manifest["generation"]["seed"]):
            raise ValueError(f"{path}: Unbekannte oder doppelte Antwortkombination {key}.")
        records[key] = result
    if metadata["total"] != len(expected_keys) or (metadata["status"] == "complete" and records.keys() != expected_keys):
        raise ValueError(f"{path}: Anzahl der Antwortkombinationen passt nicht zum Protokoll.")

    def scores(result):
        return {metric: latest.get((result["result_id"], metric), {}).get("score") if result else None
                for metric in METRICS}

    baselines = {prompt_id: records.get((prompt_id, None, 0.0)) for prompt_id in prompt_ids}
    baseline_tokens = {prompt_id: answer["output_token_ids"] for prompt_id, answer in baselines.items() if answer}
    comparisons = {}
    for layer, alpha in product(layers, alphas):
        rows = {}
        for prompt_id in prompt_ids:
            baseline, answer = baselines[prompt_id], records.get((prompt_id, layer, alpha))
            base_scores, steered_scores = scores(baseline), scores(answer)
            rows[prompt_id] = {
                "baseline_id": baseline["result_id"] if baseline else None,
                "steered_id": answer["result_id"] if answer else None,
                "baseline": base_scores, "steered": steered_scores,
                "delta": {metric: steered_scores[metric] - base_scores[metric]
                          if None not in (steered_scores[metric], base_scores[metric]) else None
                          for metric in METRICS},
            }
        comparisons[layer, alpha] = rows
    return Sample(condition, metadata, vectors, norms, vector_set.metadata["vector_hash"],
                  comparisons, baseline_tokens, len(results),
                  sum(result["total_time"] for result in results))


def stats(values):
    values = [value for value in values if value is not None]
    return {"n": len(values), "mean": mean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def cosine_row(a, b, layer):
    x, y = a.vectors[layer], b.vectors[layer]
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    cosine = float(np.clip(x @ y / denominator, -1, 1)) if denominator else None
    shared = set(a.condition["claim_ids"]) & set(b.condition["claim_ids"])
    return {"runs": [a.metadata["run_id"], b.metadata["run_id"]], "cosine": cosine,
            "shared_claims": len(shared)}


def shared_reviews(rows, metric):
    return [prompt_id for prompt_id in rows[0]
            if all(row[prompt_id]["delta"][metric] is not None for row in rows)]


def summarize_effect(samples, layer, alpha, metric):
    rows = [sample.comparisons[layer, alpha] for sample in samples]
    common = shared_reviews(rows, metric)
    return {"prompts": common, "paired_reviews_by_run": {
                sample.metadata["run_id"]: sum(entry["delta"][metric] is not None for entry in row.values())
                for sample, row in zip(samples, rows, strict=True)
            },
            **{key: stats([mean(row[prompt_id][key][metric] for prompt_id in common) for row in rows]
                          if common else [])
               for key in ("baseline", "steered", "delta")}}


def compare_topics(samples, layers, alphas):
    first, second = TOPICS
    lookup = {(sample.condition["group"], sample.condition["seed"]): sample for sample in samples}
    seeds = sorted(a_seed for topic, a_seed in lookup if topic == first and (second, a_seed) in lookup)
    if not seeds:
        return None
    pairs = [(lookup[first, seed], lookup[second, seed]) for seed in seeds]
    geometry, effects = [], []
    for layer in layers:
        cross_topic = [{"seed": seed, **cosine_row(a, b, layer)}
                       for seed, (a, b) in zip(seeds, pairs, strict=True)]
        geometry.append({"layer": layer, "pairs": cross_topic,
                         "cosine": stats(row["cosine"] for row in cross_topic)})
        for alpha in alphas:
            rows = [(a.comparisons[layer, alpha], b.comparisons[layer, alpha]) for a, b in pairs]
            metrics = {}
            for metric in METRICS:
                common = shared_reviews([row for pair in rows for row in pair], metric)
                differences = [
                    mean(a[prompt]["delta"][metric] - b[prompt]["delta"][metric] for prompt in common)
                    for a, b in rows
                ] if common else []
                metrics[metric] = {"prompts": common, "difference": stats(differences)}
            effects.append({"layer": layer, "alpha": alpha, "metrics": metrics})
    return {"first": first, "second": second, "seeds": seeds, "geometry": geometry, "effects": effects}


def analyze(manifest, paths):
    if manifest["protocol_version"] != PROTOCOL_VERSION or manifest["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("Unbekannte Protokoll- oder Rubrikversion.")
    conditions = {condition["name"]: condition for condition in manifest["conditions"]}
    samples = {}
    environment = None
    for path in paths:
        metadata = read_metadata(path)
        name = metadata.get("dataset_name")
        if name not in conditions or name in samples:
            raise ValueError(f"{path}: Unbekannte oder bereits ausgewertete Stichprobe {name}.")
        signature = {key: metadata.get(key) for key in
                     ("model_hash", "backend_fingerprint", "gguf_file_type", "system_info",
                      "python_version", "os", "git_commit")}
        if environment is not None and signature != environment:
            raise ValueError(f"{path}: Modell, Backend, Hardware oder Code-Stand weichen ab.")
        environment = signature
        samples[name] = load_sample(path, metadata, conditions[name], manifest)
    if not samples:
        raise ValueError("Mindestens ein Run-Verzeichnis ist erforderlich.")
    samples = {name: samples[name] for name in conditions if name in samples}

    reference = next((s for s in samples.values() if s.condition["group"] == "reference"), None)
    planned = defaultdict(list)
    for condition in conditions.values():
        planned[condition["group"], condition["size"]].append(condition["name"])
    groups = []
    for (group, size), names in planned.items():
        members = [samples[name] for name in names if name in samples]
        if not members:
            continue
        geometry, effects = [], []
        for layer in manifest["layers"]:
            pairs = [cosine_row(a, b, layer) for a, b in combinations(members, 2)]
            geometry.append({"layer": layer, "l2": stats([sample.norms[layer] for sample in members]),
                             "cosines": pairs, "cosine": stats([row["cosine"] for row in pairs]),
                             "vs_reference": [cosine_row(sample, reference, layer) for sample in members]
                             if reference is not None and group != "reference" else []})
            for alpha in manifest["alphas"]:
                effects.append({"layer": layer, "alpha": alpha, "metrics": {
                    metric: summarize_effect(members, layer, alpha, metric) for metric in METRICS
                }})
        groups.append({"group": group, "size": size, "planned_runs": len(names),
                       "available_runs": len(members), "geometry": geometry, "effects": effects})

    runs = []
    for name in conditions:
        if name not in samples:
            continue
        sample = samples[name]
        metadata = sample.metadata
        runs.append({**sample.condition, "run_id": metadata["run_id"], "model_hash": metadata["model_hash"],
                     "status": metadata["status"], "error": metadata.get("error"), "generated": sample.generated,
                     "planned_generations": metadata["total"], "generation_seconds": sample.generation_seconds,
                     "raw_l2": sample.norms, "vector_hash": sample.vector_hash, "conditions": [
                         {"layer": layer, "alpha": alpha, "intervention_l2": abs(alpha) * sample.norms[layer],
                          "prompts": rows} for (layer, alpha), rows in sample.comparisons.items()
                     ]})
    baseline_differences = []
    for a, b in combinations(samples.values(), 2):
        different = [prompt_id for prompt_id, tokens in a.baseline_tokens.items()
                     if prompt_id in b.baseline_tokens and tokens != b.baseline_tokens[prompt_id]]
        if different:
            baseline_differences.append({"runs": [a.metadata["run_id"], b.metadata["run_id"]], "prompts": different})
    return {"protocol_version": PROTOCOL_VERSION, "pool_hash": manifest["pool_hash"],
            "prompt_set_hash": manifest["prompt_set_hash"], "rubric_version": RUBRIC_VERSION,
            "environment": environment, "runs": runs, "groups": groups,
            "topic_comparison": compare_topics(samples.values(), manifest["layers"], manifest["alphas"]),
            "baseline_differences": baseline_differences,
            "missing_datasets": [name for name in conditions if name not in samples]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True, help="Neue JSON-Datei für die Auswertung.")
    args = parser.parse_args()
    try:
        report = analyze(read_json(args.manifest), args.runs)
        content = json_bytes(report)
        with args.output.open("xb") as file:
            file.write(content)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Auswertung in {args.output}")


if __name__ == "__main__":
    main()
