"""Menschenlesbare Konfiguration, bei jeder Benutzung frisch von der Platte."""

from copy import deepcopy
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
KINDS = ("chat", "research", "steering")
GENERATION_FIELDS = {
    "model_path", "n_gpu_layers", "n_threads", "context_size", "temperature",
    "top_k", "top_p", "min_p", "max_output_tokens", "seed", "repeat_penalty",
    "enable_thinking",
}
DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class ConfigError(ValueError):
    """Ungültige oder nicht lesbare Konfiguration mit einer erklärenden Meldung."""


def _integer(value, label, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum} bis {maximum}" if maximum is not None else f"mindestens {minimum}"
        raise ConfigError(f"{label}: erwartet wird eine ganze Zahl ({bounds}).")
    return value


def _number(value, label, minimum=None, maximum=None):
    if type(value) not in (int, float):
        raise ConfigError(f"{label}: erwartet wird eine endliche Zahl.")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ConfigError(f"{label}: erwartet wird eine endliche Zahl.") from None
    if not math.isfinite(number):
        raise ConfigError(f"{label}: NaN und unendliche Werte sind nicht erlaubt.")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{label}: der Wert muss mindestens {minimum} sein.")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{label}: der Wert darf höchstens {maximum} sein.")
    return number


def _fields(table, required, label, optional=()):
    if not isinstance(table, dict):
        raise ConfigError(f"{label}: erwartet wird eine TOML-Tabelle.")
    missing = required - table.keys()
    if missing:
        raise ConfigError(f"{label}: Pflichtfelder fehlen: {', '.join(sorted(missing))}.")
    unknown = table.keys() - required - set(optional)
    if unknown:
        raise ConfigError(f"{label}: unbekannte Felder: {', '.join(sorted(unknown))}.")


def _dataset_name(value, label, allow_empty=True):
    if not isinstance(value, str) or (value and not DATASET_NAME.fullmatch(value)):
        raise ConfigError(f"{label}: Datensatzname nur aus Buchstaben, Ziffern, '_' und '-'.")
    if not value and not allow_empty:
        raise ConfigError(f"{label}: bitte einen Datensatz auswählen.")


def _chat_layer(value):
    """Chat-Layer: einzelne Zahl oder nicht leere Liste, erst ab 1."""
    if type(value) is int:
        _integer(value, "steering.layer", 1)
        return
    if isinstance(value, (list, tuple)) and value:
        seen = set()
        for layer in value:
            _integer(layer, "steering.layer", 1)
            if layer in seen:
                raise ConfigError("steering.layer: doppelte Layer sind nicht erlaubt.")
            seen.add(layer)
        return
    raise ConfigError("steering.layer: erwartet wird eine Zahl ab 1 oder eine nicht leere Liste.")


def parse_layers(spec, layer_count: int) -> list[int]:
    """1-basierte Layer: all, Anfang-Ende[:Schritt] oder explizite Integerliste."""
    _integer(layer_count, "Layeranzahl des Modells", 1)
    if isinstance(spec, str):
        spec = spec.strip()
        if spec == "all":
            return list(range(1, layer_count + 1))
        match = re.fullmatch(r"([1-9][0-9]*)\s*-\s*([1-9][0-9]*)(?:\s*:\s*([1-9][0-9]*))?", spec)
        if not match:
            raise ConfigError("layers: erlaubt sind 'all', '1-64', '1-64:4' oder [8, 16, 24].")
        start, end = int(match[1]), int(match[2])
        step = int(match[3] or 1)
        if start > end:
            raise ConfigError("layers: der Bereichsanfang darf nicht größer als das Ende sein.")
        if start > layer_count or end > layer_count:
            raise ConfigError(f"layers: alle Layer müssen zwischen 1 und {layer_count} liegen.")
        return list(range(start, end + 1, step))
    if not isinstance(spec, (list, tuple)) or not spec:
        raise ConfigError("layers: erwartet wird eine nicht leere Layerliste oder ein Bereich.")
    for layer in spec:
        _integer(layer, "layers", 1, layer_count)
    return sorted(set(spec))


def parse_alphas(values) -> list[float]:
    """Endliche, eindeutige Alphas; Null gehört ausschließlich zur Baseline."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ConfigError("alphas: erwartet wird eine nicht leere Liste von Zahlen.")
    alphas = []
    for value in values:
        alpha = _number(value, "alphas")
        if alpha != 0 and alpha not in alphas:
            alphas.append(alpha)
    return alphas


def _generation(config):
    if not isinstance(config["model_path"], str) or not config["model_path"].strip():
        raise ConfigError("model_path: erwartet wird ein nicht leerer Modelldateipfad.")
    gpu = config["n_gpu_layers"]
    if gpu != "auto":
        _integer(gpu, "n_gpu_layers", -1)
    _integer(config["n_threads"], "n_threads", 0)
    context = _integer(config["context_size"], "context_size", 2)
    _integer(config["max_output_tokens"], "max_output_tokens", 1, context - 1)
    _integer(config["seed"], "seed", 0, 2**32 - 2)
    _integer(config["top_k"], "top_k", 0)
    _number(config["temperature"], "temperature", 0)
    top_p = _number(config["top_p"], "top_p", 0, 1)
    if top_p == 0:
        raise ConfigError("top_p: der Wert muss größer als 0 sein.")
    _number(config["min_p"], "min_p", 0, 1)
    penalty = _number(config["repeat_penalty"], "repeat_penalty", 0)
    if penalty == 0:
        raise ConfigError("repeat_penalty: der Wert muss größer als 0 sein.")
    if type(config["enable_thinking"]) is not bool:
        raise ConfigError("enable_thinking: erwartet wird true oder false.")


def validate_config(kind: str, config: dict) -> dict:
    """Validiert ohne Änderungen am Aufrufer; Modelldetails werden später geprüft."""
    if kind not in KINDS:
        raise ConfigError(f"Unbekannte Config '{kind}'. Erlaubt: {', '.join(KINDS)}.")
    if kind == "steering":
        _fields(config, {"token_position", "normalization", "cache"}, "steering")
        if config["token_position"] != "last_non_padding":
            raise ConfigError("token_position: unterstützt wird nur 'last_non_padding'.")
        if config["normalization"] not in ("unit_l2", "none"):
            raise ConfigError("normalization: erlaubt sind 'unit_l2' und 'none'.")
        if type(config["cache"]) is not bool:
            raise ConfigError("cache: erwartet wird true oder false.")
    elif kind == "chat":
        _fields(config, GENERATION_FIELDS | {"steering"}, "chat")
        _generation(config)
        steering = config["steering"]
        _fields(steering, {"enabled", "dataset", "layer", "alpha"}, "chat.steering")
        if type(steering["enabled"]) is not bool:
            raise ConfigError("steering.enabled: erwartet wird true oder false.")
        _dataset_name(steering["dataset"], "steering.dataset", not steering["enabled"])
        _chat_layer(steering["layer"])
        _number(steering["alpha"], "steering.alpha")
    else:
        _fields(config, GENERATION_FIELDS | {"prompt", "prompt_set", "layers", "alphas", "steering_dataset"}, "research")
        _generation(config)
        if not isinstance(config["prompt"], str):
            raise ConfigError("prompt: erwartet wird Text, auch ein leerer Text ist erlaubt.")
        _dataset_name(config["prompt_set"], "prompt_set")
        _dataset_name(config["steering_dataset"], "steering_dataset")
        # Der tatsächliche obere Grenzwert folgt aus den GGUF-Metadaten.
        # Die reine Syntaxprüfung benötigt keine geladene Modellinstanz.
        spec = config["layers"]
        if isinstance(spec, (list, tuple)):
            if not spec:
                raise ConfigError("layers: die Layerliste darf nicht leer sein.")
            for layer in spec:
                _integer(layer, "layers", 1)
        elif isinstance(spec, str) and spec.strip() == "all":
            pass
        elif isinstance(spec, str):
            match = re.fullmatch(r"([1-9][0-9]*)\s*-\s*([1-9][0-9]*)(?:\s*:\s*([1-9][0-9]*))?", spec.strip())
            if not match:
                raise ConfigError("layers: erlaubt sind 'all', '1-64', '1-64:4' oder [8, 16, 24].")
            if int(match[1]) > int(match[2]):
                raise ConfigError("layers: der Bereichsanfang darf nicht größer als das Ende sein.")
        else:
            raise ConfigError("layers: erwartet wird 'all', ein Bereich oder eine Layerliste.")
        parse_alphas(config["alphas"])
    return deepcopy(config)


def _config_path(kind, path):
    if kind not in KINDS:
        raise ConfigError(f"Unbekannte Config '{kind}'.")
    return Path(path) if path is not None else CONFIG_DIR / f"{kind}.toml"


def load_config(kind: str, path: Path | None = None) -> dict:
    """Hot-Reload: keine globale Config und kein Cache; jeder Aufruf liest erneut."""
    path = _config_path(kind, path)
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        return validate_config(kind, config)
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError(f"{path}: {error}") from error


def save_config(kind: str, changes: dict, path: Path | None = None) -> dict:
    """Kommentare erhalten, gesamten neuen Zustand prüfen, dann atomar ersetzen."""
    import tomlkit

    path = _config_path(kind, path)
    if not isinstance(changes, dict):
        raise ConfigError("Config-Änderungen müssen eine Tabelle sein.")
    try:
        document = tomlkit.parse(path.read_text(encoding="utf-8"))

        def update(table, values):
            for key, value in values.items():
                if isinstance(value, dict):
                    if key not in table:
                        table[key] = tomlkit.table()
                    if not isinstance(table[key], dict):
                        raise ConfigError(f"{key}: erwartet wird eine TOML-Tabelle.")
                    update(table[key], value)
                else:
                    table[key] = value

        update(document, changes)
        validated = validate_config(kind, document.unwrap())
        encoded = tomlkit.dumps(document).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return validated
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ConfigError(f"{path}: {error}") from error
