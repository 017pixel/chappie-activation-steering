"""Validated prompt lists for reproducible research runs."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tomllib

from .config import DATASET_NAME, ROOT


PROMPT_DIR = ROOT / "data" / "prompts"
PROMPT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class PromptSetError(ValueError):
    """Invalid prompt set file or name."""


@dataclass(frozen=True)
class PromptSet:
    path: Path
    source: str
    hash: str
    name: str
    description: str
    prompts: list[dict[str, str]]


def _text(table, key, label):
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PromptSetError(f"{label}.{key}: erwartet wird ein nicht leerer Text.")
    return value


def _path(name_or_path, directory):
    directory = Path(directory) if directory is not None else PROMPT_DIR
    if isinstance(name_or_path, Path):
        path = name_or_path
    elif isinstance(name_or_path, str):
        name = name_or_path.removesuffix(".toml")
        if not DATASET_NAME.fullmatch(name):
            raise PromptSetError("Prompt-Set-Name nur aus Buchstaben, Ziffern, '_' und '-'; keine Pfadbestandteile.")
        path = directory / f"{name}.toml"
        if not path.resolve().is_relative_to(directory.resolve()):
            raise PromptSetError(f"Prompt-Set-Pfad verlässt das Verzeichnis: {path}.")
    else:
        raise PromptSetError("Prompt-Set muss als Name oder pathlib.Path angegeben werden.")
    if path.suffix != ".toml":
        raise PromptSetError(f"Prompt-Sets müssen eine .toml-Datei sein: {path}.")
    if path.name.startswith("_"):
        raise PromptSetError(f"Vorlagen sind keine Prompt-Sets: {path.name}.")
    return path.resolve()


def load_prompt_set(name_or_path: str | Path, directory: Path | None = None) -> PromptSet:
    path = _path(name_or_path, directory)
    try:
        source = path.read_bytes().decode("utf-8")
        content = tomllib.loads(source)
        name = _text(content, "name", "prompt_set")
        if not DATASET_NAME.fullmatch(name):
            raise PromptSetError("prompt_set.name: nur Buchstaben, Ziffern, '_' und '-' sind erlaubt.")
        if isinstance(name_or_path, str) and name != path.stem:
            raise PromptSetError(f"prompt_set.name '{name}' muss dem Dateistem '{path.stem}' entsprechen.")
        description = _text(content, "description", "prompt_set")
        entries = content.get("prompts")
        if not isinstance(entries, list) or not entries:
            raise PromptSetError("prompts: mindestens ein [[prompts]] ist erforderlich.")
        prompts = []
        seen = set()
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise PromptSetError(f"prompts[{index}]: erwartet wird eine TOML-Tabelle.")
            prompt_id = _text(entry, "id", f"prompts[{index}]")
            if not PROMPT_ID.fullmatch(prompt_id):
                raise PromptSetError(f"prompts[{index}].id: nur Buchstaben, Ziffern, '_' und '-' sind erlaubt.")
            if prompt_id in seen:
                raise PromptSetError(f"prompts[{index}].id: doppelte ID '{prompt_id}'.")
            seen.add(prompt_id)
            prompts.append({
                "id": prompt_id,
                "category": _text(entry, "category", f"prompts[{index}]"),
                "pressure": _text(entry, "pressure", f"prompts[{index}]"),
                "prompt": _text(entry, "prompt", f"prompts[{index}]"),
                "correct_answer": _text(entry, "correct_answer", f"prompts[{index}]"),
            })
        return PromptSet(path=path, source=source, hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                         name=name, description=description, prompts=prompts)
    except (OSError, UnicodeError, ValueError) as error:
        raise PromptSetError(f"{path}: {error}") from error


def discover_prompt_sets(directory: Path | None = None) -> tuple[list[PromptSet], dict[str, str]]:
    directory = Path(directory) if directory is not None else PROMPT_DIR
    prompt_sets = []
    errors = {}
    by_name = {}
    duplicated = set()
    if not directory.is_dir():
        return [], {str(directory): "Prompt-Set-Verzeichnis fehlt oder ist kein Verzeichnis."}
    for path in sorted(directory.glob("*.toml")):
        if path.name.startswith("_"):
            continue
        try:
            prompt_set = load_prompt_set(path)
            if prompt_set.name in by_name:
                first = by_name[prompt_set.name]
                message = f"Doppelter Prompt-Set-Name '{prompt_set.name}': {first.path.name}, {path.name}."
                errors[first.path.name] = message
                errors[path.name] = message
                duplicated.add(prompt_set.name)
            else:
                by_name[prompt_set.name] = prompt_set
                prompt_sets.append(prompt_set)
            if prompt_set.name != path.stem and path.name not in errors:
                errors[path.name] = f"prompt_set.name '{prompt_set.name}' muss dem Dateistem '{path.stem}' entsprechen."
        except PromptSetError as error:
            errors[path.name] = str(error)
    return [item for item in prompt_sets if item.name not in duplicated and item.path.name not in errors], errors
