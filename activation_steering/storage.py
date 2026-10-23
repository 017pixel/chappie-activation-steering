"""Durable, file-based experiment storage. Raw JSONL records stay append-only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


class StorageError(ValueError):
    """An experiment artifact cannot be read without ambiguity."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"Ungültiger JSON-Zahlenwert: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Doppelter JSON-Schlüssel: {key}")
        result[key] = value
    return result


def parse_json(data: str | bytes) -> Any:
    return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_object)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return  # Windows does not allow opening directories via os.open.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, data: bytes) -> None:
    """Replace a derived artifact atomically, including a durable directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(Path(path), json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = parse_json(Path(path).read_bytes())
    except (OSError, ValueError, UnicodeError) as error:
        raise StorageError(f"{path}: JSON kann nicht gelesen werden: {error}") from error
    if not isinstance(value, dict):
        raise StorageError(f"{path}: Erwartet wird ein JSON-Objekt.")
    return value


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict[str, Any]]:
    """Recover only a torn final record; preserve its bytes in a separate artifact."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise StorageError(f"{path}: {error}") from error
    records: list[dict[str, Any]] = []
    lines = data.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = parse_json(line)
            if not isinstance(record, dict):
                raise ValueError("Ein JSONL-Eintrag muss ein Objekt sein.")
        except (ValueError, UnicodeError) as error:
            # A newline-terminated malformed record is not an interrupted write.
            torn_tail = index == len(lines) - 1 and not line.endswith(b"\n")
            if not repair_tail or not torn_tail:
                raise StorageError(f"{path}, Zeile {index + 1}: Ungültiger JSONL-Eintrag: {error}") from error
            backup = path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex}.bin")
            atomic_bytes(backup, data[offset:])
            atomic_bytes(path, data[:offset])
            break
        records.append(record)
        offset += len(line)
    return records


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append and fsync one record. Callers serialize concurrent writers."""
    path = Path(path)
    data = json_bytes(value)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        size = os.fstat(descriptor).st_size
        if size:
            os.lseek(descriptor, -1, os.SEEK_END)
            if os.read(descriptor, 1) != b"\n":
                data = b"\n" + data
        pending = memoryview(data)
        while pending:
            count = os.write(descriptor, pending)
            if count == 0:
                raise OSError("JSONL-Eintrag konnte nicht vollständig geschrieben werden.")
            pending = pending[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
