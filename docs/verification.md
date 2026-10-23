# Verifikation der Forschungsplattform 0.1.0

Stand: 2026-10-07. Grundlage ist der vollständige Plan
`plans/Grundstrucktur_datenset_cli_steering.md` mit 73 Abschnitten.
Die Belege betreffen Implementierung und technische Funktion, keine wissenschaftlichen Ergebnisse.

## Deterministische Prüfungen

Die Suite enthält 72 bestandene Tests. Die geprüften Bereiche sind:

- Konfiguration: alle fünf Layerformate, Duplikate, Grenzen, NaN/Infinity,
  Pflichtfelder, Tippfehler, Hot-Reload, Freeze und kommentartreues atomisches Speichern.
- Datensätze: Discovery ohne Registry, ignorierte Vorlage, exakter Hash einschließlich
  Zeilenumbrüchen, doppelte Namen, fehlende Paare, leere Texte, ungültiges TOML und Pfadprüfung.
- Vektoren: genaue paarweise Differenz, Mittelwert, ursprüngliche L2-Norm,
  Normierung, Nullvektorfehler, Cache-Invalidierung und NPZ-Integrität ohne Pickle.
- Backend-Brücke: UI-/nativer Layerindex, Pufferoffsets, erste/letzte Decoderposition,
  letzter Extraktionstoken, Addition zu allen Tokenzeilen und MTP-Ausschluss.
- Runs: 385 Vorschau, einmalige Baseline und Alpha 0, mehrere interne Prompts,
  exakte Snapshots, tatsächliche Offload-/Threadwerte, dauerhafte Speicherung,
  Lock-Konkurrenz, explizite Stale-Recovery, Unterbrechung, fehlende Kombinationen
  bei Resume, beschädigter Tail mit Backup, vollständige beschädigte Zeilen,
  unvollständiges gültiges JSON, widersprüchliche IDs und veränderte Artefakte/Seeds/Backends.
- Bewertungen: getrennte append-only Historie, einzelne und offene Metriken,
  geänderte und zurückgenommene Scores, Rubrikgrenzen und unveränderte Rohdaten.
- Terminal: Haupt-/Untermenüs, Streaming/Thinking/Verlauf, Kontextgrenze,
  `/new`-Reload einschließlich sequenziellem Modellwechsel, Config-Editor,
  Reviewfilter und Überspringen, Matrix mit acht Metriken, beiden Paginationachsen,
  separater Baseline, Teilbewertungszahlen, Detailöffnung und kleiner kompletter
  CLI-Runner-Ablauf mit ausdrücklich simulierten Testantworten in temporären Verzeichnissen.

Ausgeführt mit Exit-Code 0:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q activation_steering cli start.py tests
.venv/bin/python -m pip check
git diff --check
```

Zusätzlich wurde Ruff 0.16.10 aus einem temporären Prüfverzeichnis verwendet,
ohne eine zusätzliche Runtime-Abhängigkeit einzutragen. Die projektspezifische
Prüfung gegen Syntaxfehler, undefinierte Namen und ungenutzte Imports besteht:

```bash
PYTHONPATH=/tmp/chappie-checks .venv/bin/python -m ruff check --isolated --select E4,E7,E9,F activation_steering cli tests start.py
```

Der echte Bootstrapper wurde mit dem Projektinterpreter gestartet; Hauptmenü,
Chat-Untermenü, Forschungsmenü, Datensatzerkennung, Validierung, Templateanzeige
und Rückkehr/Beenden wurden über Terminaleingaben geprüft. Kein Modell wurde dabei
vorzeitig geladen. Gitignore-Prüfungen bestätigen ignorierte Gewichte/Caches/Locks
und versionierbare Run-Configs, JSONL, Metadaten, Vektorsnapshots und run.log.

## Echter GGUF-Smoke-Test

`tests/model_smoke.py` wurde ausdrücklich mit dem vorhandenen Qwen3.8-27B IQ4_XS
GGUF und llama-cpp-python 0.3.36 ausgeführt. Die finalen technischen Belege stehen in
`runs/2026-10-07_004_friendly_demo/verification.json`, `terminal_verification.txt`
und `terminal_startup_verification.txt`. Die früheren erfolgreichen technischen
Runs 001 und 002 bleiben ebenfalls erhalten.
Jeder erfolgreich abgeschlossene technische Run enthält sieben acht-Token-Rohantworten: eine Baseline und Layer
1/32/64 mit Alpha -1/+1. `research_use=false`, keine menschlichen Scores.

Nachweis:

- 64 echte Decoder-Residualvektoren mit je 5120 Dimensionen und L2-Norm 1.
- GGUF-Blöcke 65 minus ein zusätzlicher MTP-Block ergeben tatsächlich 64 Sweep-Layer.
- Ein zweiter Backendaufruf wurde vor dem Laden einer weiteren Gewichtskopie abgelehnt.
- Unterbrechung nach der bereits dauerhaft gespeicherten Baseline, anschließend
  Resume mit genau sechs fehlenden Kombinationen. Die Baselinebytes bleiben unverändert.
  Die finale Probe liefert zusätzlich ein echtes SIGINT während des Backend-Aufrufs;
  dieser Abbruch speichert keine unvollständige Antwort und wird sauber fortgesetzt.
- Alpha-0-Kontrolle für Layer 1, 32 und 64: identische Token-IDs für Baseline,
  Infrastruktur mit Alpha 0 und anschließende Baseline.
- Nicht-null Steuerung verändert echte Logits in allen drei geprüften Layern.
  Anschließend sind ungesteuerte Logits exakt identisch zur ursprünglichen Baseline.
- Sichtbarer Streaming-Chat ohne Steering, `/new`, erneutes Laden der Config und
  Chat mit Steering wurden mit demselben echten Backend geprüft.
- Run wird in History entdeckt. Antwortdetail, Preview, Review-Inbox und alle
  acht Matrixmetriken rendern mit echten Daten. evaluations.jsonl bleibt leer;
  unreviewed Scores werden nicht erfunden und erscheinen als ?.

Die maximalen Logitdifferenzen der initialen technischen Probe bei Alpha +1 waren
1.7786943912506104 (Layer 1), 1.1436843872070312 (32) und 0.07211971282958984 (64).
Die finale Wiederholung speichert ihre tatsächlichen Logitdifferenzen ebenfalls
in verification.json und bestätigt in allen drei Layern den Eingriff sowie
den exakten anschließenden Reset. Dies bestätigt einen tatsächlichen
numerischen Eingriff unter diesen Bedingungen.
Es bewertet weder Antwortqualität noch erwünschte Steering-Wirkung. Acht Tokens
können vor dem Ende des Thinking-Abschnitts abbrechen.

Der Versuch `2026-10-07_003_friendly_demo` machte einen Fehlerpfad sichtbar:
Die upstream-Logging-Callback-Grenze verschluckte die normale Python-SIGINT-
Exception. Dieser Testlauf bleibt mit Status failed und gültiger gespeicherter
Rohantwort erhalten. Die Korrektur merkt SIGINT während nativer Generierungs-
und Extraktionsaufrufe vor, stoppt den nächsten Graphabschnitt und gibt erst
anschließend in Python KeyboardInterrupt weiter. Der ursprüngliche Signalhandler
wird wiederhergestellt. Ein gesonderter Regressionstest prüft diese Grenze.

## Unabhängige Prüfung und Grenzen

Ein separater Luna-Verifier hat den Plan gegen Code und Dokumentation geprüft und
die 70 Tests im ersten Review selbst ausgeführt. Sein erster Review fand keinen bestätigten
Code-Blocker. Der finale Luna-Review hat Run 004 mit echtem SIGINT, die Rohdateien und die
Signalbehandlung unabhängig geprüft und alle 72 Tests erneut ausgeführt.
Ergebnis: keine bestätigten offenen Punkte, Abschluss-Gate bestanden.

Es wurde kein vollständiges 385-Experiment gestartet und kein Commit/Push ausgeführt.
Es gibt keine automatische Evaluation, keinen LLM-Judge, keine Web-UI und keine
wissenschaftlichen Aussagen über optimale Layer, Alphas oder Refusal/Safety.
Die konkrete Backend-Brücke und der echte Smoke-Test wurden auf diesem Linux-
Forschungsserver geprüft. Andere Betriebssysteme und Backend-Versionen bleiben
ungeprüft; Versionsabweichungen werden ausdrücklich abgelehnt.

## Betroffene Dateien

Neu für diesen Auftrag: `cli/main.py`, `research.py`, `review.py`, `matrix.py`,
`common.py`; `activation_steering/backend.py`, `config.py`, `datasets.py`,
`vectors.py`, `runs.py`, `storage.py`, `evaluation.py`, `locking.py`;
drei `configs/*.toml`; `data/steering/_template.toml` und `friendly_demo.toml`;
vier deterministische Testmodule und `tests/model_smoke.py`;
`docs/methodology.md`, dieser Nachweis und `diary/2026-10-07.md`.
Die technischen Demo-Runs sind bewusst Teil der nachvollziehbaren Historie.

Verändert: `start.py`, `cli/chat.py`, `requirements.txt`, `.gitignore`,
`README.md`, `AGENTS.md` und `CHANGELOG.md`. Bereits vorhandene lokale Änderungen,
der Implementierungsplan und der Tagebucheintrag vom 2026-10-06 bleiben erhalten.
