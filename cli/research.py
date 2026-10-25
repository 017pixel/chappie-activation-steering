"""Konfigurieren, starten, fortsetzen und untersuchen einfacher Forschungsruns."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from activation_steering.config import ROOT, load_config
from activation_steering.datasets import discover_datasets, load_dataset
from activation_steering.prompts import load_prompt_set
from activation_steering.runs import (
    ExperimentLock, LockError, create_run, discover_runs, execute_run,
    plan_run, read_metadata, read_snapshot_config,
)
from cli.common import (
    choose_dataset, choose_run, confirm, menu, say, settings_menu, show_config, show_json,
)


def multiline_prompt():
    say("Kein Forschungs-Prompt definiert. Prompt eingeben, /done beendet, /cancel bricht ab.")
    lines = []
    while True:
        line = input("> ")
        if line.strip() == "/cancel":
            return None
        if line.strip() == "/done":
            text = "\n".join(lines)
            if text.strip():
                return text
            say("Der Prompt darf nicht leer sein.")
            continue
        lines.append(line)


@contextmanager
def experiment_lock(run_id="preparing"):
    """Nur ein nachweislich veralteter Lock bietet eine bestätigte Recovery an."""
    lock = ExperimentLock(run_id=run_id)
    try:
        lock.__enter__()
    except LockError as error:
        say(error)
        if "Veralteter Lock:" not in str(error) or not confirm("Veralteten Lock archivieren und fortfahren?"):
            raise
        lock = ExperimentLock(run_id=run_id, recover=True)
        lock.__enter__()
    try:
        yield lock
    finally:
        lock.__exit__(None, None, None)


def preview_run(config, dataset, backend, steering_config, prompt_set=None):
    plan = plan_run(config, backend.layer_count)
    say("\nExperiment", style="bold")
    for index, prompt in enumerate(plan["prompts"]):
        say(f"Prompt {plan['prompt_ids'][index]}:\n{prompt}")
    if prompt_set is not None:
        say(f"Prompt-Set: {prompt_set.name} | SHA-256: {prompt_set.hash} | {len(prompt_set.prompts)} Prompts")
    say(f"Steering Dataset: {dataset.name} | SHA-256: {dataset.hash}")
    say(f"Target: {dataset.target_label} | Contrast: {dataset.contrast_label}")
    say(f"Paare: {len(dataset.pairs)} | research_use = {str(dataset.research_use).lower()}")
    say(f"Modell: {backend.metadata.get('model_name', backend.metadata.get('model', '?'))}")
    say(f"Modelldatei: {config['model_path']}")
    say(f"Modell-Layer: {backend.layer_count} | gewählte Layer: {plan['layers']}")
    say("Alphas: " + ", ".join(f"{alpha:+g}" for alpha in plan["alphas"]))
    say("Alpha > 0: Target. Alpha < 0: Contrast. Alpha 0 ist die gemeinsame Baseline.")
    show_config(config)
    say("\nSteering-Extraktion")
    show_config(steering_config)
    say(f"Baseline: {plan['baseline']}\nSteering Runs: {plan['steering']}\nGesamt: {plan['total']}")
    return plan


def progress(values):
    layer = "Baseline" if values["layer"] is None else f"Layer {values['layer']}"
    elapsed = values["elapsed"]
    eta = values.get("eta")
    say(f"[{values['completed']} / {values['total']}] {values['run_id']} | {layer} | "
        f"Alpha {values['alpha']:+g} | verstrichen {elapsed:.1f}s | "
        + (f"ETA {eta:.1f}s" if eta is not None else "ETA noch nicht verfügbar"))


def start_run():
    from activation_steering.backend import Backend
    from activation_steering.vectors import build_vectors
    # Der Snapshot gilt ab dieser Aktion. Kein späterer Config-Read im Runner.
    config = load_config("research")
    steering_config = load_config("steering")
    prompt_set = None
    if config["prompt_set"]:
        prompt_set = load_prompt_set(config["prompt_set"])
        config["prompts"] = [entry["prompt"] for entry in prompt_set.prompts]
        config["prompt_ids"] = [entry["id"] for entry in prompt_set.prompts]
    elif not config["prompt"].strip():
        prompt = multiline_prompt()
        if prompt is None:
            return
        config["prompt"] = prompt
    dataset = load_dataset(config["steering_dataset"]) if config["steering_dataset"] else choose_dataset()
    if dataset is None:
        return
    config["steering_dataset"] = dataset.name
    with experiment_lock() as lock:
        say("Lade Modell für die validierte Run-Vorschau ...")
        with Backend(config) as backend:
            preview_run(config, dataset, backend, steering_config, prompt_set=prompt_set)
            if not confirm("Experiment starten?"):
                say("Experiment verworfen. Kein Run angelegt.")
                return
            say("Steering-Vektoren laden / erzeugen ...")
            vector_set = build_vectors(backend, dataset, steering_config)
            run_dir = create_run(config, dataset, backend, vector_set, prompt_set=prompt_set)
            lock.update_run(run_dir.name)
            status = execute_run(run_dir, backend, on_progress=progress, lock=lock)
            say(f"Run {run_dir.name}: {status}. Rohdaten sofort gespeichert.")


def resume_run(run_dir):
    from activation_steering.backend import Backend
    metadata = read_metadata(run_dir)
    if metadata["status"] == "complete":
        say("Dieser Lauf ist bereits vollständig.")
        return
    snapshot = read_snapshot_config(run_dir)
    config = deepcopy(snapshot)
    config.pop("resolved", None)
    recorded = metadata.get("backend_metadata", {})
    for key in ("n_gpu_layers", "n_threads"):
        if recorded.get(key) is not None:
            config[key] = recorded[key]
    say(f"Fortsetzen: {run_dir.name}. Ausschließlich eingefrorene Run-Snapshots verwenden.")
    say(f"Bisher: {metadata.get('completed', '?')} / {metadata['total']} | Status: {metadata['status']}")
    if not confirm("Fehlende Generierungen fortsetzen?"):
        return
    with experiment_lock(run_dir.name) as lock:
        with Backend(config) as backend:
            status = execute_run(run_dir, backend, on_progress=progress, lock=lock)
            say(f"Run {run_dir.name}: {status}.")


def datasets_menu():
    while True:
        choice = menu("Steering-Datensätze", [("1", "Datensätze anzeigen"), ("2", "Datensatz öffnen"),
                                               ("3", "Datensatz validieren"), ("4", "Template-Pfad"),
                                               ("5", "Zurück")])
        if choice == "1":
            datasets, errors = discover_datasets()
            for dataset in datasets:
                say(f"{dataset.name}: {dataset.description}\n  {len(dataset.pairs)} Paare, "
                    f"{dataset.target_label} / {dataset.contrast_label}, "
                    f"research_use={str(dataset.research_use).lower()}\n  SHA-256: {dataset.hash}")
            for path, error in errors.items():
                say(f"{path}: {error}")
            if not datasets and not errors:
                say("Noch keine Datensätze. Vorlage kopieren und bearbeiten.")
        elif choice in ("2", "3"):
            raw = input("Datensatzname oder TOML-Pfad, q zurück: ").strip()
            if raw.lower() == "q":
                continue
            try:
                dataset = load_dataset(Path(raw) if "/" in raw else raw)
                say(f"Gültig: {dataset.name}, {len(dataset.pairs)} Paare, SHA-256 {dataset.hash}")
                if choice == "2":
                    say(dataset.source)
            except (ValueError, OSError) as error:
                say(f"Datensatz ungültig: {error}")
        elif choice == "4":
            say(ROOT / "data" / "steering" / "_template.toml")
            say("Template wird bei der Datensatzerkennung ignoriert.")
        elif choice in ("5", "q"):
            return


def history():
    from cli.matrix import matrix_view
    from cli.review import browse_answers, review_inbox
    paths = discover_runs()
    for path in paths:
        try:
            metadata = read_metadata(path)
            say(f"{path.name} | {metadata.get('dataset_name', '?')} | "
                f"{len(metadata.get('layers', []))} Layer × {len(metadata.get('alphas', []))} Alpha | "
                f"{metadata.get('completed', '?')}/{metadata['total']} | {metadata['status']}")
        except (ValueError, OSError) as error:
            say(f"{path.name}: {error}")
    run_dir = choose_run(paths)
    if run_dir is None:
        return
    while True:
        metadata = read_metadata(run_dir)
        entries = [("1", "Config"), ("2", "Metadata"), ("3", "Matrix"),
                   ("4", "Antworten"), ("5", "Review Status / Inbox")]
        if metadata["status"] != "complete":
            entries.append(("6", "Resume"))
        entries.append(("7", "Zurück"))
        choice = menu(run_dir.name, entries)
        if choice == "1":
            say((run_dir / "config.toml").read_text(encoding="utf-8"))
        elif choice == "2":
            show_json(metadata)
            from activation_steering.storage import read_json
            show_json(read_json(run_dir / "steering_metadata.json"))
        elif choice == "3":
            matrix_view(run_dir)
        elif choice == "4":
            browse_answers(run_dir)
        elif choice == "5":
            review_inbox(run_dir)
        elif choice == "6" and metadata["status"] != "complete":
            resume_run(run_dir)
        elif choice in ("7", "q"):
            return


def alpha_zero_control():
    from activation_steering.backend import Backend
    from activation_steering.vectors import build_vectors
    config = load_config("research")
    config.update(prompt="Antworte kurz: Was ist 2 plus 2?", temperature=0.0, seed=42, max_output_tokens=16)
    dataset = load_dataset("friendly_demo")
    layer = int(input("Layer für Alpha-0-Kontrolltest (1-basiert, Enter: 32): ").strip() or "32")
    if layer < 1:
        raise ValueError("Layer muss mindestens 1 sein.")
    say(f"Technischer Alpha-0-Kontrolltest mit harmlosem friendly_demo, Layer {layer}, Temperatur 0, Seed 42, 16 Tokens.")
    say("Drei Generierungen: Baseline, Alpha 0 und Baseline danach. Kein Forschungsrun.")
    if not confirm("Kontrolltest starten?"):
        return
    with experiment_lock("alpha-zero-control"):
        with Backend(config) as backend:
            if layer > backend.layer_count:
                raise ValueError(f"Layer darf höchstens {backend.layer_count} sein.")
            vectors = build_vectors(backend, dataset, load_config("steering"))
            result = backend.alpha_zero_control(config, vectors.vectors[layer], layer)
            show_json(result)


def research_menu():
    from cli.matrix import matrix_view
    from cli.review import review_inbox
    while True:
        choice = menu("Forschung", [("1", "Einstellungen"),
                                    ("2", "Forschungslauf starten"), ("3", "Aktiver / unterbrochener Lauf"),
                                    ("4", "History"), ("5", "Review Inbox"), ("6", "Matrix / Auswertung"),
                                    ("7", "Steering-Datensätze"), ("8", "Zurück"),
                                    ("9", "Technischer Alpha-0-Kontrolltest")])
        try:
            if choice == "1":
                settings_menu("research")
            elif choice == "2":
                start_run()
            elif choice == "3":
                paths = [path for path in discover_runs() if read_metadata(path)["status"] != "complete"]
                run_dir = choose_run(paths)
                if run_dir:
                    resume_run(run_dir)
            elif choice == "4":
                history()
            elif choice in ("5", "6"):
                run_dir = choose_run(discover_runs())
                if run_dir:
                    (review_inbox if choice == "5" else matrix_view)(run_dir)
            elif choice == "7":
                datasets_menu()
            elif choice in ("8", "q"):
                return
            elif choice == "9":
                alpha_zero_control()
        except KeyboardInterrupt:
            say("\nAktion unterbrochen. Bereits gespeicherte Ergebnisse bleiben erhalten.")
        except (ValueError, RuntimeError, OSError) as error:
            say(f"Forschungsaktion nicht möglich: {error}")
