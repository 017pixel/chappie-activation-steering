# Changelog

## [0.3.1] - 2026-10-23

### Erstellt
- Visualisierte Forschungsplanung für die Jugend-forscht-Strategie und den ersten Studienzuschnitt.

### Verändert
- Projektstruktur dokumentiert den Planungsordner.
- Lokale Browser-Artefakte bleiben aus dem Repository.

## [0.3.0] - 2026-10-10

### Erstellt
- Modellwechsel im Hauptmenü: alle GGUF-Dateien aus `models/` werden mit Größe aufgelistet und der Modellpfad wird in Chat- und Forschungs-Config gemeinsam gesetzt.
- Uncensored-Vergleichsmodell (Qwen3.8-27B HauhauCS Aggressive, IQ4_XS) für Alignment- und Refusal-Vergleiche eingerichtet.

### Verändert
- Die Einstellung `model_path` bietet die lokale Modellliste statt Freitext an; ein Wechsel gilt für neue Chats und Runs.
- Tests der Terminalabläufe hängen nicht mehr von der lokal gewählten Steering-Einstellung in `configs/chat.toml` ab.
- README und Projektregeln beschreiben Modellwechsel und Vergleichsmodell.

## [0.2.0] - 2026-10-08

### Erstellt
- Ein aufgeräumter Einstellungsbereich für Chat und Forschung, gegliedert in Modell & Sampling, Steering und Vektorberechnung mit Leerzeilen zwischen den Abschnitten.
- Datensatz-Auswahl als nummerierte Liste aller gültigen Dateien aus `data/steering/` statt Freitext.
- Pfeil-hoch/runter-History im Chat für die Nachrichten der laufenden Sitzung.

### Verändert
- Doppelte Steering-Konfiguration entfernt: das getrennte Menü im Chat und die separate Steering-Extraktionsoption der Forschung sind in die Einstellungen integriert.
- Werte werden in TOML-Schreibweise angezeigt und ebenso angenommen; `true`/`false`, `an`/`aus` und Layer wie `1, 2, 3` funktionieren zuverlässig.
- Beim Aktivieren von Steering fragt die Eingabe direkt nach dem Datensatz, statt mit einer Fehlermeldung abzubrechen.
- Steering an/aus hat eine eigene Eingabe (`an`/`aus`); ein leeres Enter bestätigt mit „Unverändert.“ statt still nichts zu tun.
- Der Chatstart zeigt den Steering-Status eindeutig als aktiv oder als Baseline; die Verwechslung von `research_use` mit einem Steering-Schalter ist in README und Oberfläche erklärt.
- Chat-Standard zurück auf deaktiviertes Steering mit einzelnem Layer 32.

## [0.1.0] - 2026-10-07

### Erstellt
- Kommentierte Chat-, Forschungs- und Steering-Konfigurationen mit Hot-Reload und kommentartreuer Bearbeitung im Terminal.
- Kontrastive TOML-Datensätze mit automatischer Erkennung, Validierung, Hash, Vorlage und harmlosem Demo-Kontrast.
- Echte Residualextraktion für alle 64 Decoder-Layer, explizite Vektornormalisierung und reproduzierbarer Vektorcache.
- Activation Steering über native Control Vectors und eine kleine, versionsgeprüfte Brücke für den ersten Decoder-Layer.
- Forschungsrunner mit gemeinsamer Baseline, Layer-/Alpha-Sweep, sofortiger Rohdatenspeicherung, Snapshots, Locks, Unterbrechung und Resume.
- Run-History, menschliche Review Inbox mit getrennten Bewertungshistorien und paginierte Layer-/Alpha-Matrix für Scores und Performance.
- Technischer Alpha-0-Kontrolltest und deterministische Tests für Konfiguration, Datensätze, Speicherung, Reviews und Terminalabläufe.
- Ctrl+C wird während nativer Aufrufe vorgemerkt und sicher als Unterbrechung weitergegeben, einschließlich fremder Logging-Callbacks; ein echter Generierungsabbruch prüft den Fehlerpfad.
- Methodik-Dokumentation und heutiger Tagebucheintrag zum Video und geplanten Forschungsdesign.

### Verändert
- Zentraler Einstieg öffnet jetzt das Hauptmenü mit Chat und Forschung; Streaming, Chatbefehle und Performanceanzeigen bleiben erhalten.
- Decoder-Layeranzahl wird aus GGUF-Metadaten unter Ausschluss des zusätzlichen MTP-Blocks ermittelt.
- README und Projektregeln beschreiben den terminalbasierten, dateibasierten Forschungsaufbau.
- Generierte Caches und lokale Locks werden gezielt ignoriert; Forschungsrun-Snapshots und Rohdaten bleiben versionierbar.

## [0.0.2] - 2026-10-06

### Erstellt
- Grundlegende Repository-Struktur angelegt (`cli/`, `data/`, `activation_steering/`)
- Bestehendes CLI nach `cli/chat.py` verschoben, Modellpfad robust über Repository-Root
- `data/` für zukünftige Forschungsdaten angelegt (noch keine Datensätze)
- `activation_steering/` als zukünftiges Forschungsmodul angelegt (noch keine Logik)
- Zentralen `start.py`-Bootstrapper hinzugefügt (`.venv`-Handling, Dependencies nur bei Bedarf, CUDA-Backend geschützt, Projekt startet über `python start.py`)
- Minimale `requirements.txt` erstellt (ohne GPU-Backend)

### Verändert
- README mit Startanleitung und Strukturübersicht angepasst, CLI-Pfade aktualisiert
- AGENTS.md an die neue Struktur angepasst

## [0.0.1] - 2026-10-06

### Erstellt
- Neues Repository für CHAPPiE Activation Steering angelegt, bewusst getrennt von der alten CHAPPiE-Agentenarchitektur
- Forschungsfokus auf Activation Steering festgelegt und Forschungsfrage definiert
- Qwen3.8-27B als Hauptmodell gewählt, IQ4_XS-Version heruntergeladen und lokales Modell von Git ausgeschlossen
- Einfache CLI zum direkten Chatten mit dem Basismodell erstellt (Kontextlimit 8192 Tokens, Performance-Messung TTFT und Tokens/s, Basisbefehle `/new`, `/status`, `/exit`)
- README als erste Projektbeschreibung erstellt
