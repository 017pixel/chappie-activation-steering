# CHAPPiE Activation Steering — Projektkontext für KI-Agenten

## 1. Was dieses Projekt ist

`CHAPPiE Activation Steering` ist ein experimentelles Forschungsprojekt.

Es untersucht ausschließlich:

`Activation Steering bei lokalen LLMs`

Keine Agentenarchitektur.
Kein Memory.
Keine Life-Simulation.
Keine Tools.
Kein RAG.
Keine autonome Agentenlogik.
Kein komplexer Harness.
Keine Multi-Agent-Systeme.
Keine Web-UI, außer sie dient direkt der Forschungsfrage.

Activation Steering ist der Forschungsgegenstand. Neue Implementierungen müssen einen
direkten Bezug zur Forschungsfrage haben (siehe Abschnitt 4).

## 2. Warum es existiert und woher die Idee kommt

Das Projekt ist eine neue, stark fokussierte Weiterentwicklung des früheren Projekts CHAPPiE.

Die frühere CHAPPiE-Version war wesentlich breiter aufgebaut und beschäftigte sich unter
anderem mit Cognitive-Agent-Architekturen, Memory, Life-Simulation, Persönlichkeit,
Harnesses und weiteren Agentenfunktionen.

Mit der vorherigen Version von CHAPPiE wurde am Bundeswettbewerb Künstliche Intelligenz NRW
teilgenommen. Das Projekt hat es nicht ins Finale geschafft.

Feedback und eigene Auswertung haben dazu geführt, den Forschungsgegenstand wesentlich klarer
einzugrenzen: aus dem breiten Agentenprojekt wurde ein sauberes experimentelles
Forschungsprojekt zu Activation Steering.

## 3. Langfristiges Ziel

Teilnahme an:

`Jugend forscht 2027`

Das Projekt soll dafür von Anfang an wissenschaftlich nachvollziehbar, reproduzierbar und
sehr gut dokumentiert werden. Später sollen aus Entwicklung, Experimenten und Erkenntnissen
der Forschungsbericht und die Jugend-forscht-Dokumentation entstehen.

Aktueller Status: frühe Forschungs- und Aufbauphase. Es liegen noch keine
Forschungsergebnisse vor.

## 4. Forschungsfrage

Exakt so verwenden:

> Wie wirken Layerposition und relative Steering-Stärke einer Sycophancy-Richtung auf die Bereitschaft eines lokalen 27B-LLMs, falschen Nutzeraussagen zu widersprechen – und welche Nebenwirkungen entstehen für Antwortqualität und Sicherheit?

## 5. Forschungsinteressen

Das sind Forschungsrichtungen, keine bereits bestätigten Ergebnisse:

- Wie verändert eine Sycophancy-Richtung die Bereitschaft, falschen Nutzeraussagen zu widersprechen?
- Welche Rolle spielt der Layer, in dem Steering angewendet wird?
- Welche Rolle spielt die relative Stärke des Steering-Vektors?
- Wie verändert sich die Antwortqualität?
- Welche sicherheitsrelevanten Nebenwirkungen entstehen?
- Wird das Modell bei negativer Stärke ehrlicher oder nur streitsüchtiger?
- Unterscheiden sich Originalmodell und Uncensored-Finetune?
- Wie ähnlich sind die Sycophancy-Vektoren der beiden Modelle?
- Wie verhält sich die Richtung bei Fakten-, Meinungs- und Beratungsfragen?

## 6. Aktuelles Modell und aktueller Code-Stand

- Modell: `Qwen3.8-27B`
- Quantisierung: `IQ4_XS`
- GGUF-Datei: `models/Qwen3.8-27B-IQ4_XS.gguf`
- Quelle: `bartowski/Qwen3.8-27B-GGUF` (siehe README.md)
- Vergleichsmodell: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` (HauhauCS, siehe README.md)
- Ausführung: lokal, Modellgewichte werden von Git ignoriert (`models/`, `*.gguf`)
- Kontextlimit: `8192 Tokens`

Weitere GGUF-Varianten können in `models/` liegen. Der Hauptmenüpunkt
`Modell wechseln` listet sie mit Größe auf und setzt den Pfad in beiden Configs;
die Einstellung `model_path` bietet dieselbe Liste an. Ein Wechsel gilt für neue
Chats und Runs, nie für laufende. Der Vektorcache folgt dem Modellhash.

Das terminalbasierte Haupt-CLI startet über `python start.py` und öffnet
`cli/main.py` mit Chat, Forschung und Modellwechsel. Rich genügt; keine Web-UI,
API oder Datenbank.
Das GPU-Backend wird nie automatisch ersetzt. Die Steering-Brücke unterstützt
llama-cpp-python 0.3.36 und dessen geprüftes Tensorlayout.

Chat behält Streaming, `/new`, `/status`, `/exit`, `/help` und Performanceanzeigen.
`configs/chat.toml`, `research.toml` und `steering.toml` werden vor neuen
Unterhaltungen beziehungsweise Runs neu gelesen. Laufende Bedingungen bleiben
konstant. CLI-Änderungen erhalten TOML-Kommentare.

Steering-Datensätze sind kontrastive TOML-Dateien unter `data/steering/`.
Target bedeutet positives Alpha, Contrast negatives Alpha. `_template.toml`
ist kein Datensatz; `friendly_demo` mit `research_use=false` ist nur ein Test.
Test-Prompts liegen als TOML unter `data/prompts/`; `prompt_set` in
`configs/research.toml` wählt eine Liste. Die verwendete Liste wird als
`prompt_set.toml` im Run gespeichert.
Die Brücke extrahiert echte post-FFN-Residuale und generiert einen Vektor je Layer.
Das GGUF enthält 65 Blöcke, davon einen zusätzlichen MTP-Block; der Sweep hat
64 Decoder-Layer. Semantik und Grenzen stehen in `docs/methodology.md`.

Jeder Forschungsrun enthält Config-, Dataset- und Vektor-Snapshots sowie technische
Metadaten. `results.jsonl` bleibt append-only. Menschliche Bewertungen werden
separat und historisiert in `evaluations.jsonl` angehängt; keine erfundenen Scores
und kein Selbst-Judge. Nicht bewertete Werte bleiben offen. Baseline bedeutet
vollständig deaktiviertes Steering, einmal pro Prompt. Locks erlauben nur ein
Modell beziehungsweise Experiment; Resume verwendet ausschließlich Run-Snapshots.
Reproduzierbarkeit bedeutet dokumentierte Bedingungen, keine Garantie identischer
Ausgaben bei geänderter Hardware. Noch keine wissenschaftlichen Ergebnisse.

## 7. Arbeitsweise

Dieses Projekt wird bewusst in kleinen, nachvollziehbaren Schritten entwickelt.

Vermeide große, unnötige Implementierungen.

Bevorzuge:

- kleine Änderungen
- verständlichen Code
- wenige Abstraktionen
- klare Experimente
- reproduzierbare Ergebnisse
- transparente Dokumentation

Der Entwickler ist Forscher des Projekts und möchte verstehen, was technisch passiert.
KI-Agenten sollen keine unnötig komplizierten Systeme bauen, wenn eine einfache Lösung reicht.

Bei technischen Entscheidungen gilt:

`So einfach wie möglich, aber so korrekt wie nötig.`

Der Code soll lesbar und nachvollziehbar bleiben.

## 8. Projektkontext respektieren

Vor größeren Änderungen zuerst den vorhandenen Projektkontext lesen
(`AGENTS.md`, `README.md`, `CHANGELOG.md`, ggf. `diary/`).

Keine alten CHAPPiE-v1-Konzepte wieder einführen, nur weil sie für ein Agentenprojekt
typisch wären. Insbesondere nicht ungefragt hinzufügen:

- Memory
- Tools
- RAG
- autonome Loops
- Multi-Agent-Systeme
- Life-Simulation
- Persönlichkeitssysteme
- komplexe Harnesses
- Web-UIs
- unnötige Infrastruktur

Wenn unklar ist, ob etwas zum Forschungsfokus gehört, lieber nachfragen statt Scope
hinzuzufügen.

## 9. Dokumentation als feste Regel

Nach jeder sinnvollen Änderung prüfen, ob sie dokumentiert werden muss.

Insbesondere dokumentieren:

- neue Features
- technische Änderungen
- Änderungen an der Modellkonfiguration
- Änderungen an Experimenten
- neue Forschungsentscheidungen
- neue Hypothesen
- wichtige Fehler und deren Lösungen
- Änderungen an Datensätzen
- Änderungen an Steering-Methoden
- wichtige Benchmark-Ergebnisse
- Änderungen an Evaluationen
- neue relevante Erkenntnisse

Keine erfundenen Informationen dokumentieren. Wenn etwas unbekannt oder noch nicht
entschieden ist, entsprechend kennzeichnen.

## 10. CHANGELOG und Forschungstagebuch

### CHANGELOG.md — objektive technische Projektänderungen

Beispiele: Feature hinzugefügt, Modell gewechselt, Experiment geändert, Bug behoben,
Evaluation hinzugefügt, Konfiguration verändert.

Der `versionierungen`-Skill muss nach jeder relevanten Projektänderung verwendet werden.
Der Changelog wird dauerhaft aktuell gehalten. Jede relevante Änderung ist dort
nachvollziehbar. Nicht warten, bis ausdrücklich `Aktualisiere den Changelog` gesagt wird.
Das gehört bei relevanten Änderungen automatisch zum normalen Abschluss eines Arbeitsschritts.

Der Changelog bleibt sachlich. Keine erfundenen Versionen, Daten oder Änderungen eintragen.
Struktur nach Skill `versionierungen`: neueste Version oben, Kategorien
`Erstellt` / `Verändert` / `Gelöscht` nur bei Bedarf, keine Code-Anzeigen.

### Forschungstagebuch (`diary/`) — persönlicher Forschungs- und Lernverlauf

Beispiele: neues Konzept verstanden, Video angeschaut, Paper gelesen, Idee entwickelt,
Beobachtung gemacht, Forschungsfrage verändert, Experiment interpretiert.

Das Tagebuch wird grundsätzlich auf Anweisung des Entwicklers gepflegt.
Der Agent erzeugt dort nicht ungefragt ständig Einträge. Bei passenden Arbeitsschritten
darf er fragen, z. B.:

`Möchtest du, dass ich die heutige Erkenntnis im Forschungstagebuch dokumentiere?`

Details siehe Skill `research-diary` unter `.agents/skills/research-diary/SKILL.md`.
Dateiformat: `diary/YYYY-MM-DD.md`.

## 11. Skills

Projekt-Skills liegen agentenübergreifend unter `.agents/skills/<name>/SKILL.md`.
Dieser Pfad wird sowohl von Codex als auch von OpenCode erkannt.

Vorhanden:

- `versionierungen` (globaler Skill): Changelog und Versionierung nach jeder relevanten Änderung.
- `research-diary` (Projekt-Skill unter `.agents/skills/research-diary/`): Forschungstagebuch,
  nur auf ausdrückliche Anweisung Einträge erstellen oder ergänzen, sonst nur vorschlagen.

Kein Skill darf proprietäre Sonderlogik einführen, wenn sie nicht notwendig ist.

## 12. Repository-Struktur

- `AGENTS.md`: dieser Projektkontext
- `README.md`: Projektbeschreibung und CLI-Anleitung
- `CHANGELOG.md`: technische Änderungen, Versionen
- `start.py`: zentraler Einstiegspunkt (`python start.py`)
- `requirements.txt`: normale Dependencies (ohne GPU-Backend, siehe Datei)
- `cli/`: Hauptmenü, Chat, Forschung, History, Review und responsive Matrix
- `activation_steering/`: Config-/Datasetvalidierung, GGUF-Backend, Vektoren, Runs und menschliche Evaluation
- `configs/`: kommentierte, menschenlesbare TOML-Konfigurationen
- `data/steering/`: editierbare kontrastive Datensätze und Vorlage
- `data/prompts/`: TOML-Prompt-Listen für Forschungsruns
- `data/vectors/`: reproduzierbarer, ignorierter Vektorcache
- `runs/`: vollständige Forschungsrun-Historie; nicht pauschal ignorieren
- `docs/methodology.md`: tatsächliche Steering-Semantik und Bewertungsrubriken
- `tests/`: deterministische Tests und ausdrücklich gestarteter echter Modell-Smoke-Test
- `models/`: lokale Modelldatei (Git-ignoriert, nie committen)
- `diary/`: Forschungstagebuch, eine Datei pro Tag (`YYYY-MM-DD.md`)
- `.agents/skills/`: projekt-spezifische Skills (Codex- und OpenCode-kompatibel)
