"""Frozen experiment snapshots, durable results and explicit resume recovery."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import socket
import time
import tomllib
from typing import Any, Callable, Iterator
import uuid

from .storage import StorageError, append_jsonl, atomic_bytes, atomic_json, read_json, read_jsonl
from .locking import lock_file, unlock_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"


class RunError(ValueError):
    """A run cannot be executed or resumed safely."""


class LockError(RunError):
    """An active or unrecovered experiment owns the runtime lock."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@contextmanager
def file_guard(path: Path) -> Iterator[None]:
    """Serialize short file mutations, including stale-lock recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as guard:
        lock_file(guard)
        try:
            yield
        finally:
            unlock_file(guard)


def _process_start(pid: int) -> str | None:
    try:
        # Field 22 identifies a process even if its PID is reused.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _lock_stale(owner: dict[str, Any]) -> bool:
    if owner.get("hostname") != socket.gethostname():
        return False  # A remote process cannot be proven dead locally.
    pid = owner.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    old_start = owner.get("process_start")
    current_start = _process_start(pid)
    return old_start is not None and current_start is not None and old_start != current_start


def _recover_locked(path: Path) -> Path | None:
    if not path.exists():
        return None
    owner = read_json(path)
    if not _lock_stale(owner):
        raise LockError(f"Lock von Run {owner.get('run_id', '?')} ist aktiv oder nicht sicher als veraltet erkennbar.")
    backup = path.with_name(f"{path.name}.recovered-{uuid.uuid4().hex}.json")
    os.replace(path, backup)
    return backup


def recover_lock(runs_dir: Path | str | None = None) -> Path | None:
    """Explicitly recover a proven stale lock, keeping its original owner record."""
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    with file_guard(directory / ".experiment.lock.guard"):
        return _recover_locked(directory / ".experiment.lock")


class ExperimentLock:
    """Acquire before model loading; pass the same owner into execute_run."""

    def __init__(self, run_id: str = "preparing", runs_dir: Path | str | None = None, recover: bool = False):
        self.runs_dir = (Path(runs_dir) if runs_dir is not None else RUNS_DIR).resolve()
        self.path = self.runs_dir / ".experiment.lock"
        self.run_id = run_id
        self.recover = recover
        self.token = uuid.uuid4().hex
        self.acquired = False

    def __enter__(self) -> ExperimentLock:
        if self.acquired:
            raise LockError("Dieses Experiment-Lock ist bereits erworben.")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with file_guard(self.runs_dir / ".experiment.lock.guard"):
            if self.path.exists() and self.recover:
                _recover_locked(self.path)
            owner = {
                "run_id": self.run_id, "start_time": utc_now(), "pid": os.getpid(),
                "hostname": socket.gethostname(), "process_start": _process_start(os.getpid()),
                "token": self.token,
            }
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                current = read_json(self.path)
                hint = "Veralteter Lock: explizite Recovery erforderlich." if _lock_stale(current) else "Ein Experiment ist bereits aktiv."
                raise LockError(f"{hint} Run {current.get('run_id', '?')}, PID {current.get('pid', '?')}.") from error
            try:
                with os.fdopen(descriptor, "wb") as output:
                    from .storage import json_bytes
                    output.write(json_bytes(owner))
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                self.path.unlink(missing_ok=True)
                raise
            self.acquired = True
        return self

    def update_run(self, run_id: str) -> None:
        if not self.acquired:
            raise LockError("Experiment-Lock wurde noch nicht erworben.")
        with file_guard(self.runs_dir / ".experiment.lock.guard"):
            owner = read_json(self.path)
            if owner.get("token") != self.token:
                raise LockError("Experiment-Lock gehört einem anderen Prozess.")
            owner["run_id"] = run_id
            atomic_json(self.path, owner)
            self.run_id = run_id

    def __exit__(self, *_: Any) -> None:
        if self.acquired:
            with file_guard(self.runs_dir / ".experiment.lock.guard"):
                if self.path.exists() and read_json(self.path).get("token") == self.token:
                    self.path.unlink()
            self.acquired = False


def _prompts(config: dict[str, Any]) -> list[str]:
    prompts = config.get("prompts", [config.get("prompt", "")])
    if not isinstance(prompts, list) or not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise RunError("Für den Run wird mindestens ein nicht leerer Prompt benötigt.")
    return prompts


def _prompt_ids(config: dict[str, Any], prompt_count: int) -> list[str]:
    ids = config.get("prompt_ids")
    if ids is None:
        return [f"p{index:04d}" for index in range(1, prompt_count + 1)]
    if (not isinstance(ids, list) or len(ids) != prompt_count
            or any(not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", item) for item in ids)):
        raise RunError("prompt_ids: erwartet wird eine eindeutige ID je Prompt.")
    if len(set(ids)) != len(ids):
        raise RunError("prompt_ids: doppelte IDs sind nicht erlaubt.")
    return ids


def plan_run(config: dict[str, Any], layer_count: int) -> dict[str, Any]:
    from .config import parse_alphas, parse_layers
    layers = parse_layers(config.get("layers", "all"), layer_count)
    alphas = parse_alphas(config.get("alphas", []))
    prompts = _prompts(config)
    prompt_ids = _prompt_ids(config, len(prompts))
    baseline = len(prompts)
    steering = baseline * len(layers) * len(alphas)
    return {"layers": layers, "alphas": alphas, "prompts": prompts, "prompt_ids": prompt_ids,
            "baseline": baseline, "steering": steering, "total": baseline + steering}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(backend: Any) -> dict[str, Any]:
    names = (
        "model_hash", "model_sha256", "model_file", "model_path", "model_size", "model_size_bytes", "model_mtime_ns",
        "quantization", "chat_template", "chat_template_hash", "backend_version", "llama_cpp_version",
        "llama_cpp_python_version", "control_vector_method", "extraction_method", "context_size",
        "backend_library_hashes", "bridge_version", "steering_semantics", "architecture", "upstream_source_reference",
        "n_gpu_layers", "n_threads", "gpu", "cpu", "total_block_count", "nextn_predict_layers",
    )
    return {**{name: backend.metadata[name] for name in names if name in backend.metadata},
            "layer_count": backend.layer_count, "dimension": backend.dimension}


def create_run(config: dict[str, Any], dataset: Any, backend: Any, vector_set: Any,
               runs_dir: Path | str | None = None, prompt_set: Any = None) -> Path:
    """Freeze all scientifically relevant inputs before the first generation."""
    import tomlkit
    from .vectors import save_vectors
    plan = plan_run(config, backend.layer_count)
    if "context_size" in backend.metadata and backend.metadata["context_size"] != config.get("context_size"):
        raise RunError("Run-Kontextgröße unterscheidet sich von der geladenen Modell-Kontextgröße.")
    missing = set(plan["layers"]) - set(vector_set.vectors)
    if missing:
        raise RunError(f"Steering-Vektoren fehlen für Layer: {sorted(missing)}")
    if hashlib.sha256(dataset.source.encode("utf-8")).hexdigest() != dataset.hash:
        raise RunError("Dataset-Inhalt stimmt nicht mit seinem SHA-256-Hash überein.")
    if vector_set.metadata.get("dataset_hash") != dataset.hash:
        raise RunError("Steering-Vektoren stammen nicht aus dem ausgewählten Dataset.")
    if vector_set.metadata.get("vector_dimension") != backend.dimension:
        raise RunError("Steering-Vektordimension passt nicht zum Modell.")
    import numpy as np
    for layer in plan["layers"]:
        vector = vector_set.vectors[layer]
        if vector.shape != (backend.dimension,) or not np.isfinite(vector).all():
            raise RunError(f"Ungültiger Steering-Vektor für Layer {layer}.")
    vector_model = vector_set.metadata.get("model")
    if vector_model is not None and vector_model.get("model_hash") != backend.metadata.get("model_hash"):
        raise RunError("Steering-Vektoren stammen aus einem anderen Modell.")
    frozen_config = dict(config)
    for key in ("n_gpu_layers", "n_threads"):
        if key in backend.metadata:
            frozen_config[key] = backend.metadata[key]
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset.name).strip("_")[:48] or "dataset"
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for index in range(1, 100000):
        run_dir = directory / f"{day}_{index:03d}_{slug}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RunError("Kein freier Run-Name gefunden.")
    snapshot = {**frozen_config, "resolved": {
        "layers": plan["layers"], "alphas": plan["alphas"],
        "prompts": plan["prompts"], "prompt_ids": plan["prompt_ids"],
    }}
    atomic_bytes(run_dir / "config.toml", tomlkit.dumps(snapshot).encode("utf-8"))
    atomic_bytes(run_dir / "steering_dataset.toml", dataset.source.encode("utf-8"))
    if prompt_set is not None:
        atomic_bytes(run_dir / "prompt_set.toml", prompt_set.source.encode("utf-8"))
    save_vectors(run_dir / "vectors.npz", vector_set)
    steering_metadata = {
        **vector_set.metadata, "dataset_name": dataset.name, "dataset_hash": dataset.hash,
        "target_label": dataset.target_label, "contrast_label": dataset.contrast_label,
        "dataset_description": dataset.description, "pair_count": len(dataset.pairs),
        "research_use": dataset.research_use, "layers": sorted(vector_set.vectors),
        "dimension": backend.dimension,
    }
    atomic_json(run_dir / "steering_metadata.json", steering_metadata)
    for name in ("results.jsonl", "evaluations.jsonl", "run.log"):
        atomic_bytes(run_dir / name, b"")
    artifacts = ["config.toml", "steering_dataset.toml", "steering_metadata.json", "vectors.npz"]
    if prompt_set is not None:
        artifacts.append("prompt_set.toml")
    metadata = {
        **backend.metadata, "run_id": run_dir.name, "start_time": utc_now(), "end_time": None,
        "status": "pending", "schema_version": 1, "dataset_name": dataset.name,
        "dataset_hash": dataset.hash, "research_use": dataset.research_use,
        "prompt_set_name": prompt_set.name if prompt_set is not None else None,
        "prompt_set_hash": prompt_set.hash if prompt_set is not None else None,
        "backend_metadata": backend.metadata, "backend_fingerprint": _fingerprint(backend),
        "requested_model_settings": {key: config.get(key) for key in ("n_gpu_layers", "n_threads")},
        "layer_count": backend.layer_count, "dimension": backend.dimension,
        "layers": plan["layers"], "alphas": plan["alphas"], "total": plan["total"],
        "completed": 0, "context_size": config.get("context_size"),
        "generation_settings": {key: config.get(key) for key in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "seed", "max_output_tokens", "enable_thinking")},
        "artifact_hashes": {name: _hash(run_dir / name) for name in artifacts},
    }
    atomic_json(run_dir / "metadata.json", metadata)
    append_jsonl(run_dir / "run.log", {"at": utc_now(), "event": "created", "total": plan["total"]})
    from .evaluation import summarize
    summarize(run_dir)
    return run_dir


def read_metadata(run_dir: Path | str) -> dict[str, Any]:
    metadata = read_json(Path(run_dir) / "metadata.json")
    required = ("run_id", "status", "total", "artifact_hashes", "backend_fingerprint")
    if any(key not in metadata for key in required):
        raise RunError(f"{run_dir}: Run-Metadaten sind unvollständig.")
    if not isinstance(metadata["total"], int) or isinstance(metadata["total"], bool) or metadata["total"] < 1:
        raise RunError(f"{run_dir}: Ungültige Anzahl geplanter Generierungen.")
    if not isinstance(metadata["artifact_hashes"], dict) or not isinstance(metadata["backend_fingerprint"], dict):
        raise RunError(f"{run_dir}: Ungültige Snapshot-Metadaten.")
    return metadata


def discover_runs(runs_dir: Path | str | None = None) -> list[Path]:
    directory = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    if not directory.exists():
        return []
    return sorted((path for path in directory.iterdir() if path.is_dir() and (path / "metadata.json").is_file()), reverse=True)


def read_results(run_dir: Path | str, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    records = read_jsonl(Path(run_dir) / "results.jsonl", repair_tail=repair_tail)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        _validate_result(record, Path(run_dir) / "results.jsonl")
        result_id = record.get("result_id")
        if not isinstance(result_id, str) or not result_id:
            raise RunError(f"{run_dir}: Resultat ohne gültige Result-ID.")
        if result_id in unique and unique[result_id] != record:
            raise RunError(f"{run_dir}: Widersprüchliche Rohdaten für Result-ID {result_id}.")
        unique[result_id] = record
    return list(unique.values())


def _validate_result(record: dict[str, Any], path: Path) -> None:
    """A JSON object is complete only with a full identity and measured response."""
    def invalid(field: str) -> None:
        raise StorageError(f"{path}: Ungültige oder unvollständige Rohantwort, Feld '{field}'.")

    for key in ("result_id", "prompt_id", "prompt", "finish_reason", "created_at"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            invalid(key)
    if not isinstance(record.get("response"), str):
        invalid("response")
    for key in ("prompt_tokens", "output_tokens", "repeat_id", "seed"):
        if type(record.get(key)) is not int or record[key] < 0:
            invalid(key)
    if record["seed"] > 2**32 - 2:
        invalid("seed")
    for key in ("ttft", "generation_time", "total_time", "tokens_per_second", "alpha"):
        value = record.get(key)
        if value is None and key in ("ttft", "tokens_per_second") and key in record:
            continue
        try:
            valid_number = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid_number = False
        if not valid_number or (key != "alpha" and value < 0):
            invalid(key)
    if record["output_tokens"] > 0 and record["ttft"] is None:
        invalid("ttft")
    tokens = record.get("output_token_ids")
    if not isinstance(tokens, list) or len(tokens) != record["output_tokens"] or any(type(token) is not int or token < 0 for token in tokens):
        invalid("output_token_ids")
    if record.get("type") == "baseline":
        if record.get("layer") is not None or record["alpha"] != 0:
            invalid("baseline layer/alpha")
    elif record.get("type") == "steering":
        if type(record.get("layer")) is not int or record["layer"] < 1 or record["alpha"] == 0:
            invalid("steering layer/alpha")
    else:
        invalid("type")


def _combinations(plan: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for index, prompt in enumerate(plan["prompts"]):
        prompt_id = plan["prompt_ids"][index]
        common = {"prompt_id": prompt_id, "repeat_id": 0, "prompt": prompt}
        prefix = f"{prompt_id}_r0"
        yield {**common, "result_id": prefix + "_baseline", "type": "baseline", "layer": None, "alpha": 0.0}
        for layer in plan["layers"]:
            for alpha in plan["alphas"]:
                yield {**common, "result_id": f"{prefix}_l{layer:03d}_a{alpha:+.17g}", "type": "steering", "layer": layer, "alpha": alpha}


def read_snapshot_config(run_dir: Path | str) -> dict[str, Any]:
    try:
        with (Path(run_dir) / "config.toml").open("rb") as source:
            config = tomllib.load(source)
    except (OSError, ValueError) as error:
        raise RunError(f"Config-Snapshot kann nicht gelesen werden: {error}") from error
    return config


def _validate_snapshot(run_dir: Path, backend: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    from .vectors import load_vectors
    metadata = read_metadata(run_dir)
    expected_artifacts = {"config.toml", "steering_dataset.toml", "steering_metadata.json", "vectors.npz"}
    if metadata.get("prompt_set_name"):
        expected_artifacts.add("prompt_set.toml")
    if set(metadata["artifact_hashes"]) != expected_artifacts:
        raise RunError("Run-Snapshot enthält keine vollständigen Integritäts-Hashes.")
    for name, digest in metadata["artifact_hashes"].items():
        try:
            actual = _hash(run_dir / name)
        except OSError as error:
            raise RunError(f"Snapshot-Artefakt fehlt: {name}") from error
        if actual != digest:
            raise RunError(f"Snapshot-Artefakt wurde verändert: {name}. Resume abgelehnt.")
    if _fingerprint(backend) != metadata["backend_fingerprint"]:
        raise RunError("Modell, Chat-Template oder Backend unterscheidet sich vom Run-Snapshot. Resume abgelehnt.")
    config = read_snapshot_config(run_dir)
    resolved = config.pop("resolved", None)
    plan = plan_run(config, backend.layer_count)
    keys = ("layers", "alphas", "prompts")
    if isinstance(resolved, dict) and "prompt_ids" in resolved:
        keys += ("prompt_ids",)
    if not isinstance(resolved, dict) or any(resolved.get(key) != plan[key] for key in keys):
        raise RunError("Aufgelöste Run-Konfiguration ist inkonsistent.")
    if metadata["total"] != plan["total"]:
        raise RunError("Geplante Run-Anzahl ist inkonsistent.")
    vector_set = load_vectors(run_dir / "vectors.npz")
    if vector_set.metadata.get("dataset_hash") != metadata["dataset_hash"]:
        raise RunError("Vector-Dataset-Hash passt nicht zum Run.")
    import numpy as np
    for layer in plan["layers"]:
        vector = vector_set.vectors.get(layer)
        if vector is None or vector.shape != (backend.dimension,) or not np.isfinite(vector).all():
            raise RunError(f"Ungültiger Steering-Vektor für Layer {layer}.")
    return metadata, config, vector_set


def execute_run(run_dir: Path | str, backend: Any, on_progress: Callable[[dict[str, Any]], None] | None = None,
                lock: ExperimentLock | None = None) -> str:
    """Resume exclusively from immutable snapshots; generate missing cells only."""
    run_dir = Path(run_dir).resolve()
    if lock is None:
        with ExperimentLock(run_dir.name, run_dir.parent) as owned_lock:
            return execute_run(run_dir, backend, on_progress, lock=owned_lock)
    if not lock.acquired or lock.runs_dir != run_dir.parent:
        raise LockError("Für den Run fehlt ein gültiger Experiment-Lock.")
    try:
        return _execute_locked(run_dir, backend, on_progress, lock)
    finally:
        reset = getattr(backend, "reset_steering", None)
        if callable(reset):
            reset()


def _execute_locked(run_dir: Path, backend: Any, on_progress: Callable[[dict[str, Any]], None] | None,
                    lock: ExperimentLock) -> str:
    lock.update_run(run_dir.name)
    metadata, config, vector_set = _validate_snapshot(run_dir, backend)
    plan = plan_run(config, backend.layer_count)
    expected = {record["result_id"]: record for record in _combinations(plan)}
    results = read_results(run_dir, repair_tail=True)
    completed: set[str] = set()
    for result in results:
        result_id = result["result_id"]
        identity = expected.get(result_id)
        if identity is None or any(result.get(key) != value for key, value in identity.items()):
            raise RunError(f"Rohdaten enthalten eine unbekannte oder abweichende Kombination: {result_id}.")
        if result["seed"] != config["seed"] or result["output_tokens"] > config["max_output_tokens"] or result["prompt_tokens"] + result["output_tokens"] > config["context_size"]:
            raise RunError(f"Gespeicherte Generation widerspricht eingefrorenem Seed oder Tokenlimit: {result_id}.")
        completed.add(result_id)
    started = time.monotonic()
    metadata.update(status="running", end_time=None, completed=len(completed), resumed_at=utc_now())
    metadata.pop("error", None)
    atomic_json(run_dir / "metadata.json", metadata)
    append_jsonl(run_dir / "run.log", {"at": utc_now(), "event": "running", "completed": len(completed)})
    try:
        for result_id, combination in expected.items():
            if result_id in completed:
                continue
            elapsed = time.monotonic() - started
            progress = {"run_id": run_dir.name, "completed": len(completed), "total": plan["total"],
                        "layer": combination["layer"], "alpha": combination["alpha"], "elapsed": elapsed,
                        "eta": None, "result_id": result_id}
            if on_progress is not None:
                on_progress(progress)
            generated = backend.generate(
                [{"role": "user", "content": combination["prompt"]}], config,
                vector=vector_set.vectors[combination["layer"]] if combination["type"] == "steering" else None,
                layer=combination["layer"], alpha=combination["alpha"],
            )
            if not isinstance(generated, dict) or not isinstance(generated.get("response"), str):
                raise RunError("Backend lieferte keine gültige Rohantwort.")
            result = {**generated, **combination, "created_at": utc_now()}
            _validate_result(result, run_dir / "results.jsonl")
            if result["seed"] != config["seed"] or result["output_tokens"] > config["max_output_tokens"] or result["prompt_tokens"] + result["output_tokens"] > config["context_size"]:
                raise RunError("Backend-Generation widerspricht dem eingefrorenen Seed oder Tokenlimit.")
            append_jsonl(run_dir / "results.jsonl", result)
            completed.add(result_id)
            metadata["completed"] = len(completed)
            atomic_json(run_dir / "metadata.json", metadata)
            elapsed = time.monotonic() - started
            progress.update(completed=len(completed), elapsed=elapsed)
            generated_count = len(completed) - len(results)
            if generated_count:
                progress["eta"] = elapsed / generated_count * (plan["total"] - len(completed))
            if on_progress is not None:
                on_progress(progress)
        status = "complete"
    except KeyboardInterrupt:
        completed = {record["result_id"] for record in read_results(run_dir, repair_tail=True)}
        status = "interrupted"
    except BaseException as error:
        completed = {record["result_id"] for record in read_results(run_dir, repair_tail=True)}
        metadata.update(status="failed", end_time=utc_now(), completed=len(completed), error=str(error))
        atomic_json(run_dir / "metadata.json", metadata)
        append_jsonl(run_dir / "run.log", {"at": utc_now(), "event": "failed", "error": str(error)})
        from .evaluation import summarize
        summarize(run_dir)
        raise
    metadata.update(status=status, end_time=utc_now(), completed=len(completed))
    atomic_json(run_dir / "metadata.json", metadata)
    append_jsonl(run_dir / "run.log", {"at": utc_now(), "event": status, "completed": len(completed)})
    from .evaluation import summarize
    summarize(run_dir)
    return status
