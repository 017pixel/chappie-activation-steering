"""Kleine gemeinsame Hilfen für die Rich-Terminaloberfläche."""

import json
from pathlib import Path
import re
import tomllib

from rich.console import Console
from rich.table import Table
from rich.text import Text

from activation_steering.config import ROOT, load_config, save_config
from activation_steering.datasets import discover_datasets

console = Console(highlight=False)
MODELS_DIR = ROOT / "models"


def say(value="", *, style=None, end="\n"):
    # Prompts und Antworten dürfen keine Rich-Markup-Tags ausführen.
    console.print(str(value), markup=False, highlight=False, style=style, end=end)


def menu(title, entries):
    say(f"\n{title}", style="bold")
    for key, label in entries:
        say(f"[{key}] {label}")
    return input("> ").strip().lower()


def confirm(label):
    return input(f"{label} [y/N] ").strip().lower() in ("y", "yes", "j", "ja")


def show_json(value):
    say(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def show_config(config):
    for key, value in config.items():
        if isinstance(value, dict):
            say(f"[{key}]")
            for name, nested in value.items():
                say(f"  {name} = {nested!r}")
        else:
            say(f"{key} = {value!r}")


def toml_literal(value):
    """Werte so anzeigen, wie sie in die TOML-Datei geschrieben werden."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_literal(item) for item in value) + "]"
    return repr(value)


def parse_layers_input(raw):
    """Versteht 32, 8,16,24, [8, 16] oder mit Leerzeichen getrennt (Chat)."""
    text = raw.strip().removeprefix("[").removesuffix("]")
    parts = [part for chunk in text.split(",") for part in chunk.split()]
    if not parts:
        raise ValueError("Mindestens ein Layer ist erforderlich.")
    layers = [int(part) for part in parts]
    if any(layer < 1 for layer in layers):
        raise ValueError("Layer müssen mindestens 1 sein.")
    if len(set(layers)) != len(layers):
        raise ValueError("Doppelte Layer sind nicht erlaubt.")
    return layers[0] if len(layers) == 1 else layers


def parse_layers_spec(raw):
    """Forschungslayer: 'all', Bereich '1-64:4' oder explizite Liste."""
    text = raw.strip()
    if text == "all" or re.fullmatch(r"[1-9][0-9]*\s*-\s*[1-9][0-9]*(?:\s*:\s*[1-9][0-9]*)?", text):
        return text
    layers = parse_layers_input(text)
    return layers if isinstance(layers, list) else [layers]


def _parse_value(raw, current, key=""):
    text = raw.strip()
    if key == "n_gpu_layers":
        try:
            return tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError:
            return text
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in ("true", "an", "on", "ja", "1"):
            return True
        if lowered in ("false", "aus", "off", "nein", "0"):
            return False
        raise ValueError("erwartet an/aus bzw. true/false")
    if isinstance(current, str):
        if text.startswith(('"', "'", "[")):
            return tomllib.loads(f"value = {text}")["value"]
        return text
    if isinstance(current, (list, tuple)):
        body = text.removeprefix("[").removesuffix("]")
        parts = [part for chunk in body.split(",") for part in chunk.split()]
        if not parts:
            raise ValueError("leere Liste")
        return [tomllib.loads(f"value = {part}")["value"] for part in parts]
    return tomllib.loads(f"value = {text}")["value"]


def choose_dataset(allow_empty=False):
    """Alle gültigen Datensätze nummeriert anbieten; None bei Abbruch."""
    datasets, errors = discover_datasets()
    for path, error in errors.items():
        say(f"Ungültiger Datensatz {path}: {error}")
    if not datasets:
        say("Keine gültigen Datensätze. Dateien in data/steering/ anlegen.")
        return None
    for index, dataset in enumerate(datasets, 1):
        note = " (Demo, research_use=false)" if not dataset.research_use else ""
        say(f"[{index}] {dataset.name}: {dataset.target_label} / {dataset.contrast_label}{note}")
    while True:
        hint = "0 leer, q zurück: " if allow_empty else "q zurück: "
        raw = input(f"Datensatznummer, {hint}").strip()
        if raw.lower() == "q":
            return None
        if allow_empty and raw == "0":
            return ""
        try:
            index = int(raw) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(datasets):
            return datasets[index]
        say("Bitte eine vorhandene Datensatznummer eingeben.")


def discover_models():
    """Alle GGUF-Dateien aus models/, alphabetisch; keine Unterordner."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted((path for path in MODELS_DIR.glob("*.gguf") if path.is_file()),
                  key=lambda path: path.name.lower())


def model_display_path(path):
    """Projektrelative Anzeige, absolute nur außerhalb des Projekts."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def choose_model(current=None):
    """Nummerierte Liste der lokalen GGUF-Dateien; relativer Pfad oder None."""
    models = discover_models()
    if not models:
        say(f"Keine GGUF-Dateien in {model_display_path(MODELS_DIR)} gefunden. Modell dort ablegen.")
        return None
    for index, path in enumerate(models, 1):
        size = path.stat().st_size / 1024**3
        note = " (aktuell)" if model_display_path(path) == current else ""
        say(f"[{index}] {path.name} ({size:.2f} GiB){note}")
    while True:
        raw = input("Modellnummer wählen, q zurück: ").strip()
        if raw.lower() == "q":
            return None
        try:
            index = int(raw) - 1
        except ValueError:
            index = -1
        if 0 <= index < len(models):
            return model_display_path(models[index])
        say("Bitte eine vorhandene Modellnummer eingeben.")


def model_menu():
    """Modell für Chat und Forschung in einem Schritt wechseln."""
    try:
        chat = load_config("chat")["model_path"]
        research = load_config("research")["model_path"]
    except (ValueError, OSError) as error:
        say(f"Modelle nicht lesbar: {error}")
        return
    say(f"Aktuell Chat: {chat}")
    say(f"Aktuell Forschung: {research}")
    selected = choose_model(chat if chat == research else None)
    if selected is None:
        return
    if selected == chat == research:
        say("Modell unverändert.")
        return
    save_config("chat", {"model_path": selected})
    save_config("research", {"model_path": selected})
    say(f"Modell gesetzt: {selected}")
    say("Gilt für neue Chats und neue Runs. Vektoren werden bei Bedarf neu berechnet.")


SETTINGS_SECTIONS = {
    "chat": (
        ("Modell & Sampling", "chat",
         ("model_path", "n_gpu_layers", "n_threads", "context_size", "temperature", "top_k",
          "top_p", "min_p", "repeat_penalty", "max_output_tokens", "seed", "enable_thinking")),
        ("Steering (Chat)", "chat",
         ("steering.enabled", "steering.dataset", "steering.layer", "steering.alpha")),
        ("Vektorberechnung (gilt auch für Forschung)", "steering",
         ("token_position", "normalization", "cache")),
    ),
    "research": (
        ("Forschungslauf", "research", ("prompt", "steering_dataset", "layers", "alphas")),
        ("Modell & Sampling", "research",
         ("model_path", "n_gpu_layers", "n_threads", "context_size", "temperature", "top_k",
          "top_p", "min_p", "repeat_penalty", "max_output_tokens", "seed", "enable_thinking")),
        ("Vektorberechnung (gilt auch für Forschung)", "steering",
         ("token_position", "normalization", "cache")),
    ),
}


def _lookup(config, key):
    value = config
    for part in key.split("."):
        value = value[part]
    return value


def settings_menu(kind):
    """Einstellungen je Bereich, in kommentierte Abschnitte gegliedert."""
    sections = SETTINGS_SECTIONS[kind]
    while True:
        try:
            configs = {name: load_config(name) for name in {file_kind for _, file_kind, _ in sections}}
        except (ValueError, OSError) as error:
            say(f"Einstellungen nicht lesbar: {error}")
            return
        available = None
        say(f"\nEinstellungen: {ROOT / 'configs' / (kind + '.toml')}", style="bold")
        fields = []
        for title, file_kind, keys in sections:
            say()
            say(title, style="bold")
            for key in keys:
                value = _lookup(configs[file_kind], key)
                fields.append((file_kind, key, value))
                suffix = ""
                if key in ("steering.dataset", "steering_dataset"):
                    if available is None:
                        available = len(discover_datasets()[0])
                    suffix = f"   ({available} verfügbar)"
                say(f"[{len(fields)}] {key} = {toml_literal(value)}{suffix}")
        raw = input("Wertnummer ändern, q zurück: ").strip()
        if raw.lower() == "q":
            return
        try:
            index = int(raw) - 1
            if not 0 <= index < len(fields):
                raise ValueError("Ungültige Wertnummer.")
            file_kind, key, current = fields[index]
            if key in ("steering.dataset", "steering_dataset"):
                dataset = choose_dataset(allow_empty=True)
                if dataset is None:
                    continue
                value = dataset if isinstance(dataset, str) else dataset.name
            elif key == "steering.enabled":
                state = "an" if current else "aus"
                text = input(f"Steering ist {state}. Neuer Wert an/aus (Enter behält): ").strip()
                if not text:
                    say("Unverändert.")
                    continue
                value = _parse_value(text, current, key)
            elif key == "steering.layer":
                value = parse_layers_input(input("Layer (1-basiert, z. B. 32 oder 1, 2, 3): "))
            elif key == "layers":
                value = parse_layers_spec(input("Layer: all, 1-64, 1-64:4 oder 1, 2, 3: "))
            elif key == "model_path":
                selected = choose_model(current)
                if selected is None:
                    continue
                value = selected
            else:
                text = input(f"{key}, neuer Wert (Enter behält, TOML): ").strip()
                if not text:
                    say("Unverändert.")
                    continue
                value = _parse_value(text, current, key)
            if key == "steering.enabled" and value and not _lookup(configs[file_kind], "steering.dataset"):
                say("Steering aktivieren: bitte zuerst einen Datensatz wählen.")
                dataset = choose_dataset()
                if dataset is None:
                    continue
                changes = {"steering": {"dataset": dataset.name, "enabled": True}}
            elif "." in key:
                parent, child = key.split(".", 1)
                changes = {parent: {child: value}}
            else:
                changes = {key: value}
            save_config(file_kind, changes)
            say("Gespeichert. Neue Werte gelten beim nächsten Chat oder Lauf.")
        except (ValueError, OSError, tomllib.TOMLDecodeError) as error:
            say(f"Einstellungen nicht geändert: {error}")


def choose_run(paths):
    if not paths:
        say("Noch keine Forschungsläufe vorhanden.")
        return None
    for index, path in enumerate(paths, 1):
        say(f"[{index}] {path.name}")
    while True:
        raw = input("Laufnummer öffnen, q zurück: ").strip()
        if raw.lower() == "q":
            return None
        try:
            index = int(raw) - 1
            if 0 <= index < len(paths):
                return paths[index]
        except ValueError:
            pass
        say("Bitte eine vorhandene Laufnummer eingeben.")


def performance(result):
    labels = {
        "prompt_tokens": "Prompt Tokens", "output_tokens": "Output Tokens",
        "ttft": "TTFT (s)", "tokens_per_second": "TPS (tok/s)",
        "generation_time": "Generation Time (s)", "total_time": "Total Time (s)",
        "seed": "Seed", "finish_reason": "Beendigung",
    }
    table = Table(show_header=False, box=None, padding=(0, 1))
    for key, label in labels.items():
        value = result.get(key)
        table.add_row(Text(label), Text(f"{value:.3f}" if isinstance(value, float)
                                        else str(value) if value is not None else "nicht verfügbar"))
    console.print(table)
