"""Zentraler Einstiegspunkt: bereitet .venv vor und startet cli/main.py.

Nur ein kleiner Bootstrapper (Linux und Windows):
    python start.py

Ablauf: Repository-Root bestimmen, .venv prüfen/erstellen, normale
Dependencies aus requirements.txt nur bei Bedarf installieren, dann das
CLI mit dem .venv-Interpreter starten.

Das GPU-Backend (llama-cpp-python, ggf. CUDA-Build) wird nie automatisch
installiert oder ersetzt. Fehlt es, gibt es eine klare Fehlermeldung
mit Hinweis auf README.md.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
CLI = ROOT / "cli" / "main.py"
MODEL_PATH = ROOT / "models" / "Qwen3.8-27B-IQ4_XS.gguf"
REQUIREMENTS = ROOT / "requirements.txt"


def fail(message):
    print(f"Fehler: {message}", file=sys.stderr)
    return 1


def venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv():
    python = venv_python()
    if VENV_DIR.is_dir():
        if python.is_file():
            return python, None
        return None, (f"{VENV_DIR} existiert, aber {python} fehlt. "
                      "Defektes Verzeichnis löschen und erneut starten.")
    print(f"Erstelle {VENV_DIR} ...")
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0 or not python.is_file():
        return None, (".venv konnte nicht erstellt werden. "
                      "Python-venv prüfen (Linux ggf. Paket python3-venv).")
    return python, None


def missing_requirements(python):
    """Pakete aus requirements.txt, deren Modul im .venv fehlt.

    Paketname wird als Modulname geprüft (klein, '-' zu '_').
    Reicht für diese kleine Liste; llama-cpp-python steht bewusst
    nicht in requirements.txt (CUDA-Schutz, siehe Dateikommentar).
    """
    missing = []
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name = re.split(r"[=<>!~\s\[]", line, 1)[0].strip().lower().replace("-", "_")
        if not name:
            continue
        check = subprocess.run([str(python), "-c", f"import {name}"],
                               capture_output=True)
        if check.returncode != 0:
            missing.append(line)
    return missing


def main():
    if sys.version_info < (3, 12):
        return fail(f"Python 3.12 oder neuer nötig (gefunden: "
                    f"{sys.version.split()[0]}).")

    python, error = ensure_venv()
    if error:
        return fail(error)

    if REQUIREMENTS.is_file():
        missing = missing_requirements(python)
        if missing:
            print(f"Installiere fehlende Dependencies: {', '.join(missing)}")
            result = subprocess.run(
                [str(python), "-m", "pip", "install",
                 "--disable-pip-version-check", "-r", str(REQUIREMENTS)])
            if result.returncode != 0:
                return fail("Dependencies konnten nicht installiert werden.")
    else:
        print("Hinweis: requirements.txt fehlt, Schritt übersprungen.")

    check = subprocess.run([str(python), "-c", "import llama_cpp"],
                           capture_output=True)
    if check.returncode != 0:
        # Gleicher CUDA-Preload wie in cli/chat.py: ohne ihn findet der
        # Loader libcudart trotz installiertem Wheel nicht. Spiegelt
        # bewusst die dortige Ladereihenfolge, statt neu zu erfinden.
        probe = (
            "import ctypes, sysconfig; from pathlib import Path; "
            "d = Path(sysconfig.get_path('platlib')) / 'nvidia/cu13/lib'; "
            "[ctypes.CDLL(str(d / n), mode=ctypes.RTLD_GLOBAL) "
            "for n in ('libcudart.so.13', 'libcublas.so.13') if (d / n).is_file()]; "
            "import llama_cpp"
        )
        check = subprocess.run([str(python), "-c", probe],
                               capture_output=True)
    if check.returncode != 0:
        return fail("llama-cpp-python fehlt im .venv. "
                    "Nicht automatisch installiert, weil je nach System ein "
                    "spezielles Wheel (CUDA/CPU) oder Build nötig ist. "
                    "Installation siehe README.md.")

    if not MODEL_PATH.is_file():
        return fail(f"Modelldatei fehlt: {MODEL_PATH}\n"
                    "Download siehe README.md.")
    if not CLI.is_file():
        return fail(f"CLI fehlt: {CLI}")

    print("Starte CLI ...", flush=True)
    result = subprocess.run([str(python), "-m", "cli.main"], cwd=str(ROOT))
    return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStart abgebrochen.")
        sys.exit(130)
