"""Streaming-Chat mit eingefrorener Konfiguration und optionalem Steering."""

from contextlib import ExitStack
from pathlib import Path
import sys

# Der bisherige Direktstart bleibt möglich: python cli/chat.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import readline
except ImportError:  # Windows ohne GNU readline
    readline = None

from activation_steering.config import load_config, save_config
from activation_steering.datasets import load_dataset
from cli.common import console, menu, performance, say, settings_menu, show_config


def assistant_message(answer, prompt):
    """Das GGUF-Template kann den öffnenden Thinking-Tag bereits enthalten."""
    reasoning, separator, content = answer.partition("</think>")
    if separator:
        reasoning = reasoning.removeprefix("<think>")
        return {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if prompt.rstrip().endswith("<think>") or answer.lstrip().startswith("<think>"):
        return {"role": "assistant", "content": "",
                "reasoning_content": answer.removeprefix("<think>")}
    return {"role": "assistant", "content": answer}


def _chat_layers(config, layer_count):
    steering = config["steering"]
    if not steering["enabled"]:
        return []
    value = steering["layer"]
    layers = [value] if type(value) is int else list(value)
    for layer in layers:
        if not 1 <= layer <= layer_count:
            raise ValueError(f"Layer muss zwischen 1 und {layer_count} liegen.")
    return layers


def _chat_vectors(backend, config):
    layers = _chat_layers(config, backend.layer_count)
    if not layers:
        say("Steering: AUS (Baseline, ein Durchlauf)")
        return {}
    from activation_steering.vectors import build_vectors
    dataset = load_dataset(config["steering"]["dataset"])
    label = layers[0] if len(layers) == 1 else layers
    say(f"Steering: {dataset.name}, Layer {label}, Alpha {config['steering']['alpha']:+g}")
    vectors = build_vectors(backend, dataset, load_config("steering")).vectors
    return {layer: vectors[layer] for layer in layers}


def chat_session():
    from activation_steering.backend import Backend
    config = load_config("chat")
    if readline is not None:
        readline.clear_history()
    say("Lade lokales Modell ...")
    with ExitStack() as resources:
        backend = resources.enter_context(Backend(config))
        vectors = _chat_vectors(backend, config)
        messages = []
        last = None
        say("Lokaler Chat. /new, /status, /thinking on|off, /exit, /help")
        while True:
            text = input("\nDu: ").strip()
            if not text:
                continue
            if text == "/exit":
                return
            if text == "/help":
                say("/new: neuer Chat und aktuelle Config | /status: Modell und Performance | /thinking on|off: Denken an/aus | /exit: zurück")
                continue
            if text.startswith("/thinking"):
                parts = text.split()
                if len(parts) == 1:
                    state = "an" if config.get("enable_thinking", True) else "aus"
                    say(f"Thinking ist {state}. Wechsel mit /thinking on|off.")
                    continue
                raw = parts[1].lower()
                if raw in ("on", "true", "1", "an"):
                    value = True
                elif raw in ("off", "false", "0", "aus"):
                    value = False
                else:
                    say("Verwende /thinking on|off.")
                    continue
                try:
                    save_config("chat", {"enable_thinking": value})
                except (ValueError, OSError) as error:
                    say(f"Thinking nicht geändert: {error}")
                    continue
                config["enable_thinking"] = value
                say(f"Thinking {'an' if value else 'aus'}. Gilt ab der nächsten Nachricht.")
                continue
            if text == "/new":
                updated = load_config("chat")
                reload_keys = ("context_size", "model_path", "n_gpu_layers", "n_threads")
                if any(updated.get(key) != config.get(key) for key in reload_keys):
                    backend.close()
                    backend = resources.enter_context(Backend(updated))
                else:
                    backend.reset()
                config = updated
                vectors = _chat_vectors(backend, config)
                messages.clear()
                last = None
                say("Neuer Chat. Aktuelle Config geladen.")
                continue
            if text == "/status":
                used = backend.model.n_tokens
                say(f"Modell: {backend.metadata.get('model_name', backend.metadata.get('model', 'nicht verfügbar'))}")
                say(f"Modelldatei: {backend.metadata.get('model_file', backend.metadata.get('model_path', 'nicht verfügbar'))}")
                say(f"Quantisierung: {backend.metadata.get('quantization', 'nicht verfügbar')}")
                say(f"Backend: llama-cpp-python {backend.metadata.get('llama_cpp_python_version', 'nicht verfügbar')}")
                say(f"Context Limit: {config['context_size']} Tokens")
                say(f"Context Tokens (Backend): {used}")
                say(f"Verbleibende Context Tokens: {config['context_size'] - used}")
                say(f"Nachrichten: {len(messages)} | Layer: {backend.layer_count}")
                say(f"GPU-Offloading-Konfiguration: {backend.metadata.get('n_gpu_layers', 'nicht verfügbar')}")
                say(f"Modell-Ladezeit: {backend.load_time:.2f} s")
                show_config(config)
                if last:
                    performance(last)
                continue
            if text.startswith("/"):
                say("Unbekannter Command. Verwende /help.")
                continue
            if readline is not None:
                readline.add_history(text)
            candidate = messages + [{"role": "user", "content": text}]
            prompt = backend.format_prompt(candidate, config.get("enable_thinking", True))
            available = config["context_size"] - len(backend.tokens(prompt)) - 1
            if available < config["max_output_tokens"]:
                say("Zu wenig Kontextplatz. Bitte /new verwenden. Nachricht nicht übernommen.")
                continue
            steering = config["steering"]
            layers = _chat_layers(config, backend.layer_count) if steering["enabled"] else []
            if layers:
                for position, layer in enumerate(layers, 1):
                    if len(layers) > 1:
                        say(f"\n--- Layer {layer} ({position}/{len(layers)}) ---")
                    else:
                        say("Modell: ", end="")
                    try:
                        last = backend.generate(
                            candidate, config, vector=vectors[layer],
                            layer=layer, alpha=steering["alpha"],
                            on_piece=lambda piece: (say(piece, end=""), console.file.flush()),
                        )
                    finally:
                        backend.reset_steering()
                    say()
                    messages = candidate + [assistant_message(last["response"], prompt)]
                    performance(last)
                if len(layers) > 1:
                    say("Alle Layer antworten auf denselben Prompt. Nur die letzte Antwort bleibt im Verlauf.")
            else:
                say("Modell: ", end="")
                try:
                    last = backend.generate(
                        candidate, config, vector=None, layer=None, alpha=0,
                        on_piece=lambda piece: (say(piece, end=""), console.file.flush()),
                    )
                finally:
                    backend.reset_steering()
                say()
                messages = candidate + [assistant_message(last["response"], prompt)]
                performance(last)
            if config["context_size"] - backend.model.n_tokens < config["max_output_tokens"]:
                say("Der Kontext ist fast voll. Bitte /new verwenden.")


def chat_menu():
    while True:
        choice = menu("Chat", [("1", "Chat starten"), ("2", "Einstellungen"), ("3", "Zurück")])
        if choice == "1":
            try:
                chat_session()
            except KeyboardInterrupt:
                say("\nChat beendet.")
            except (ValueError, RuntimeError, OSError) as error:
                say(f"Chat nicht möglich: {error}")
        elif choice == "2":
            settings_menu("chat")
        elif choice in ("3", "q"):
            return


def main():
    try:
        chat_session()
    except (KeyboardInterrupt, EOFError):
        say("\nChat beendet.")
    except (ValueError, RuntimeError, OSError) as error:
        say(f"Chat nicht möglich: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
