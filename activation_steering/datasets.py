"""Kontrastive TOML-Datensätze ohne Registry und mit exaktem Datei-Hash."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tomllib

from activation_steering.config import DATASET_NAME, ROOT


DATASET_DIR = ROOT / "data" / "steering"


class DatasetError(ValueError):
    """Fehler in einer Datensatzdatei oder ihrem Namen."""


@dataclass(frozen=True)
class Dataset:
    path: Path
    source: str
    hash: str
    name: str
    description: str
    target_label: str
    contrast_label: str
    pairs: list[dict[str, str]]
    research_use: bool


def _text(table, key, label):
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{label}.{key}: erwartet wird ein nicht leerer Text.")
    return value


def _path(name_or_path, directory):
    directory = Path(directory) if directory is not None else DATASET_DIR
    if isinstance(name_or_path, Path):
        # Explizite Path-Objekte werden auch für eingefrorene Run-Snapshots verwendet.
        path = name_or_path
    elif isinstance(name_or_path, str):
        name = name_or_path.removesuffix(".toml")
        if not DATASET_NAME.fullmatch(name):
            raise DatasetError("Datensatzname nur aus Buchstaben, Ziffern, '_' und '-'; keine Pfadbestandteile.")
        path = directory / f"{name}.toml"
        if not path.resolve().is_relative_to(directory.resolve()):
            raise DatasetError(f"Datensatzpfad verlässt das Datensatzverzeichnis: {path}.")
    else:
        raise DatasetError("Datensatz muss als Name oder pathlib.Path angegeben werden.")
    if path.suffix != ".toml":
        raise DatasetError(f"Datensätze müssen eine .toml-Datei sein: {path}.")
    if path.name.startswith("_"):
        raise DatasetError(f"Vorlagen sind keine Datensätze: {path.name}.")
    return path.resolve()


def load_dataset(name_or_path: str | Path, directory: Path | None = None) -> Dataset:
    path = _path(name_or_path, directory)
    try:
        source = path.read_bytes().decode("utf-8")
        content = tomllib.loads(source)
        name = _text(content, "name", "dataset")
        if not DATASET_NAME.fullmatch(name):
            raise DatasetError("dataset.name: nur Buchstaben, Ziffern, '_' und '-' sind erlaubt.")
        if isinstance(name_or_path, str) and name != path.stem:
            raise DatasetError(f"dataset.name '{name}' muss dem Dateistem '{path.stem}' entsprechen.")
        description = _text(content, "description", "dataset")
        target_label = _text(content, "target_label", "dataset")
        contrast_label = _text(content, "contrast_label", "dataset")
        research_use = content.get("research_use", True)
        if type(research_use) is not bool:
            raise DatasetError("research_use: erwartet wird true oder false.")
        pairs = content.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise DatasetError("pairs: mindestens ein [[pairs]] mit target und contrast ist erforderlich.")
        validated_pairs = []
        for index, pair in enumerate(pairs, start=1):
            if not isinstance(pair, dict):
                raise DatasetError(f"pairs[{index}]: erwartet wird eine TOML-Tabelle.")
            validated_pairs.append({key: _text(pair, key, f"pairs[{index}]") for key in ("target", "contrast")})
        return Dataset(
            path=path, source=source, hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            name=name, description=description, target_label=target_label,
            contrast_label=contrast_label, pairs=validated_pairs, research_use=research_use,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise DatasetError(f"{path}: {error}") from error


def discover_datasets(directory: Path | None = None) -> tuple[list[Dataset], dict[str, str]]:
    """Jeden Aufruf neu suchen; ungültige Dateien stören keine gültigen Datensätze."""
    directory = Path(directory) if directory is not None else DATASET_DIR
    datasets = []
    errors = {}
    by_name = {}
    duplicated = set()
    if not directory.is_dir():
        return [], {str(directory): "Datensatzverzeichnis fehlt oder ist kein Verzeichnis."}
    for path in sorted(directory.glob("*.toml")):
        if path.name.startswith("_"):
            continue
        try:
            if not path.resolve().is_relative_to(directory.resolve()):
                raise DatasetError("Datei verweist außerhalb des Datensatzverzeichnisses.")
            dataset = load_dataset(path)
            if dataset.name in by_name:
                first = by_name[dataset.name]
                message = f"Doppelter Datensatzname '{dataset.name}': {first.path.name}, {path.name}."
                errors[first.path.name] = message
                errors[path.name] = message
                duplicated.add(dataset.name)
            else:
                by_name[dataset.name] = dataset
                datasets.append(dataset)
            if dataset.name != path.stem and path.name not in errors:
                errors[path.name] = f"dataset.name '{dataset.name}' muss dem Dateistem '{path.stem}' entsprechen."
        except DatasetError as error:
            errors[path.name] = str(error)
    return [dataset for dataset in datasets if dataset.name not in duplicated and dataset.path.name not in errors], errors
