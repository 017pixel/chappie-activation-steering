"""Mean paired residual differences with explicit normalization and caching."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .backend import BRIDGE_VERSION, ROOT


@dataclass
class VectorSet:
    vectors: dict[int, np.ndarray]
    metadata: dict


def _digest(vectors):
    digest = hashlib.sha256()
    for layer, value in sorted(vectors.items()):
        digest.update(str(layer).encode())
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def save_vectors(path, vector_set):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(vector_set.metadata, vector_hash=_digest(vector_set.vectors), vector_format="npz-float32-v1")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary, metadata=np.array(json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
                **{f"layer_{layer}": np.asarray(value, dtype=np.float32) for layer, value in vector_set.vectors.items()})
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    vector_set.metadata.update(metadata)


def load_vectors(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            vectors = {int(name.removeprefix("layer_")): data[name].copy() for name in data.files if name.startswith("layer_")}
        if metadata["vector_format"] != "npz-float32-v1":
            raise ValueError("unbekanntes Format")
        if sorted(vectors) != metadata["layers"]:
            raise ValueError("Layerliste stimmt nicht überein")
        for layer, value in vectors.items():
            if value.dtype != np.dtype("float32") or value.shape != (metadata["vector_dimension"],) or not np.isfinite(value).all():
                raise ValueError(f"ungültiger Vektor in Layer {layer}")
        if _digest(vectors) != metadata["vector_hash"]:
            raise ValueError("Vektor-Prüfsumme stimmt nicht überein")
        return VectorSet(vectors, metadata)
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError(f"Vektorartefakt ungültig: {path}: {error}") from error


def build_vectors(backend, dataset, steering_config, cache_dir=None):
    settings = {
        "dataset_hash": dataset.hash, "model_hash": backend.metadata["model_hash"],
        "backend_version": backend.metadata["llama_cpp_python_version"],
        "backend_library_hashes": backend.metadata.get("backend_library_hashes"),
        "method_version": BRIDGE_VERSION, "token_position": steering_config["token_position"],
        "normalization": steering_config["normalization"], "layer_count": backend.layer_count,
        "vector_dimension": backend.dimension, "context_size": backend.config["context_size"],
        "n_gpu_layers": backend.metadata.get("n_gpu_layers"), "n_threads": backend.metadata.get("n_threads"),
        "text_format": "raw_text_add_bos_true_special_false_no_padding",
    }
    if settings["token_position"] != "last_non_padding" or settings["normalization"] not in ("unit_l2", "none"):
        raise ValueError("Nicht unterstützte Extraction-/Normalisierungseinstellung.")
    key = hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cache_dir = Path(cache_dir) if cache_dir is not None else ROOT / "data" / "vectors"
    cache_path = cache_dir / f"{dataset.name}_{key}.npz"
    if steering_config["cache"] and cache_path.is_file():
        result = load_vectors(cache_path)
        if result.metadata["cache_key"] != key or result.metadata["extraction_settings"] != settings:
            raise ValueError("Cache-Einstellungen passen nicht zum Vektorartefakt.")
        return result
    sums = {layer: np.zeros(backend.dimension, dtype=np.float64) for layer in range(1, backend.layer_count + 1)}
    for pair in dataset.pairs:
        target = backend.extract(pair["target"])
        contrast = backend.extract(pair["contrast"])
        if set(target) != set(sums) or set(contrast) != set(sums):
            raise ValueError("Aktivierungen enthalten nicht alle Modell-Layer.")
        for layer in sums:
            sums[layer] += target[layer].astype(np.float64) - contrast[layer].astype(np.float64)
    vectors, raw_norms = {}, {}
    for layer, total in sums.items():
        mean = total / len(dataset.pairs)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm):
            raise ValueError(f"Nicht-endlicher Steering-Vektor in Layer {layer}.")
        raw_norms[str(layer)] = norm
        if settings["normalization"] == "unit_l2":
            if norm == 0:
                raise ValueError(f"Nullvektor in Layer {layer} kann nicht auf L2-Norm 1 normalisiert werden.")
            mean = mean / norm
        vectors[layer] = mean.astype(np.float32)
    result = VectorSet(vectors, {
        "dataset_name": dataset.name, "dataset_hash": dataset.hash,
        "target_label": dataset.target_label, "contrast_label": dataset.contrast_label,
        "pair_count": len(dataset.pairs), "research_use": dataset.research_use,
        "token_position": settings["token_position"], "extraction_method": "mean_paired_post_ffn_residual_last_token",
        "normalization": settings["normalization"], "vector_dimension": backend.dimension,
        "layers": sorted(vectors), "raw_norms": raw_norms,
        "created_at": datetime.now(timezone.utc).isoformat(), "model": backend.metadata,
        "backend_version": settings["backend_version"], "cache_key": key, "extraction_settings": settings,
        "vector_hash": _digest(vectors), "vector_format": "npz-float32-v1",
    })
    if steering_config["cache"]:
        save_vectors(cache_path, result)
    return result
