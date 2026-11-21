# Quantisierungsvergleich auf der Tesla T4

Wie verändern kleinere GGUF-Quantisierungen desselben Qwen3.8-27B-Checkpoints die ungestörten Antworten, die Wirkung von Activation Steering und den Ressourcenverbrauch? Die Quantisierung wechselt zwischen Runs. Innerhalb jedes Runs bleiben Modell und Vektorsatz fest.

## Matrix und Modellquelle

Der Versuch verwendet die [Modellkarte](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/87410e7b9a5c21e6ca70c0c9cc5237de909fe621/README.md) und die [Dateiliste](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/87410e7b9a5c21e6ca70c0c9cc5237de909fe621) an derselben Revision. Die Modellkarte nennt `Qwen/Qwen3.8-27B` als Basismodell. Die GGUF-Dateihashes identifizieren die veröffentlichten Artefakte. Eine konkrete BF16-Quellrevision ist daraus nicht ableitbar.

| Gruppe | Quantisierungen in Versuchsreihenfolge |
| --- | --- |
| Referenz | IQ4_XS |
| Erste Abstufung | IQ3_M, Q3_K_M, Q3_K_S, IQ2_M, IQ2_XXS |
| Übrige kleinere Varianten | Q3_K_L, IQ3_XS, IQ3_XXS, Q2_K, IQ2_S, IQ2_XS |

Die Ausgabe von `plan` enthält Dateinamen, SHA-256 und exakte Bytezahlen. IQ4_XS hat 15,48 GB, IQ2_XXS 8,88 GB (dezimal). `Q2_K_L` hat 13,08 GB und ist in der Modellkarte als Vorgänger-Layout gekennzeichnet. Es bleibt außerhalb der Hauptmatrix und ist separat mit Prüfsumme dokumentiert. Der Uncensored-Finetune, Projektoren, die Imatrix und größere Quantisierungen gehören nicht zur Stichprobe.

Der Pilot hält den bisherigen Sycophancy-Datensatz und die acht Hauptstudienprompts per Hash fest. Layer 8, 32 und 56, Alpha −1, +1 und +2, dazu genau eine vollständig ungesteuerte Baseline pro Prompt, ergeben 80 Antworten je Quant und Profil. Die Ausgabe erfolgt bei Kontext 8192, ohne Thinking und mit Greedy-Sampling und Seed 42. Die Vektoren werden pro Quant und Run aus den Aktivierungen neu berechnet. Bei gleicher relativer Stärke kann die absolute Eingriffsnorm `|alpha| * L2` zwischen Quants variieren. Der bestehende Konstruktionsdatensatz koppelt Zustimmung an genervten Ton. Diesen möglichen Einfluss in jeder Interpretation offenlegen.

## Vorbereitung auf dem Forschungsserver

```bash
python -m activation_steering.quant_study plan
```

Der Befehl braucht Python 3.11 oder neuer und liest die Konfiguration sowie die zwei Datensätze. Er lädt weder Gewichte noch llama.cpp. Für die folgenden Schritte gilt die Projektumgebung mit `llama-cpp-python==0.3.36` und deren geprüfter Steering-Brücke.

1. Mit `df -h models` den freien Speicher und mit `nvidia-smi` die Tesla T4, den Treiber, die VRAM-Auslastung und andere GPU-Prozesse prüfen. CUDA-Runtime, Host-RAM und Swap-Status ergänzend mit den verfügbaren Systemwerkzeugen protokollieren. Für Downloads und Laufzeit messen und archivieren die Forschenden pro Versuch in einem eigenen Verzeichnis.
2. Die vorhandene Referenzdatei zuerst verwenden. Der Größen- und Hashcheck prüft ihre Zugehörigkeit zur gepinnten Revision. Bei einer Abweichung die vorhandene Datei erhalten und ihre Herkunft klären. Für die gepinnte Datei lässt sich auf demselben Dateisystem ein separates Verzeichnis wählen und die Datei von dort unter ihrem erwarteten Namen in `models/` verlinken, sobald der Name frei ist.
3. Vor jeder weiteren Variante die Modellkarte und LFS-Metadaten der gepinnten Revision prüfen. Erst nach abgeschlossenem Versuch die nächste Datei beziehen. Ein Beispiel für den optionalen Hugging-Face-CLI-Download auf dem Forschungsserver:

```bash
hf download bartowski/Qwen3.8-27B-GGUF Qwen3.8-27B-IQ3_M.gguf \
  --revision 87410e7b9a5c21e6ca70c0c9cc5237de909fe621 --local-dir models
```

Dateien und Forschungsruns bleiben bei allen Projektbefehlen erhalten. Wenn Speicher knapp wird, kann man nach dem Archivieren der Modellfingerprints einzelne neu bezogene GGUF-Dateien manuell entfernen. Die Referenz und ihre Run-Artefakte aufbewahren. Die Dateigröße allein sagt wenig über das verfügbare VRAM-Budget aus. KV-/Recurrent-Cache, Compute-Buffer und CUDA belegen zusätzlichen Speicher.

## Test und Ausführung je Variante

Für **`quality`** zuerst eine gemeinsame feste Offload-Zahl ermitteln, die mit der Referenz und allen erfolgreichen Varianten stabil funktioniert. Im Beispiel steht 32 nur als Startwert. Die tatsächlich ausgelagerten Layer aus den nativen Logs prüfen und bei abweichendem Offload den Qualitätsvergleich gesondert behandeln. Dieselbe Hardware, GPU-Auswahl, Threadzahl, Backend-Bibliothek, Cache-Konfiguration und das gleiche Chat-Template verwenden.

```bash
python -m activation_steering.quant_study run IQ4_XS --gpu-layers 32 --smoke-only \
  > iq4_xs-smoke.json 2> iq4_xs-smoke.log

python -m activation_steering.quant_study run IQ4_XS --gpu-layers 32 \
  > iq4_xs-quality.json 2> iq4_xs-quality.log
```

Jeder Aufruf prüft Dateigröße und SHA-256, GGUF-Dateityp, 64 reguläre Decoder-Layer und Vektordimension. Der Smoke-Test extrahiert mit einem kleinen technischen Datensatz echte `l_out`-Aktivierungen für alle regulären Layer. Er prüft Layer 1, 32 und 64 auf identische Baseline-Token bei Alpha 0, endliche und veränderte Logits bei Alpha +1 und den Reset zurück zu den ursprünglichen Logits. Der Smoke-Test allein erzeugt kein Forschungsrun-Verzeichnis. Der eigentliche Run führt diese Kontrollen vor der ersten Pilotgenerierung erneut aus. Rohantworten, Bewertungen und Snapshots liegen im normalen Run-Format. Unterbrochene oder pausierte Runs über die bestehende Funktion **Unterbrochene Runs** im CLI fortsetzen, sobald die identische GGUF-Datei bereitliegt.

Für **`performance`** je Quant ein separat gemessenes stabiles GPU-Profil wählen. `auto` liefert einen ersten Schätzwert. Tatsächliche Offload-Zeilen aus der nativen Ausgabe aufbewahren. Bei derselben Laufzeitumgebung beispielsweise so protokollieren:

```bash
/usr/bin/time -v -o iq3_m-time.txt \
  python -m activation_steering.quant_study run IQ3_M --profile performance --gpu-layers auto \
  > iq3_m-performance.json 2> iq3_m-performance.log
```

In einem zweiten Terminal für dieselbe Messsession den VRAM-Verbrauch der gewählten GPU mit `nvidia-smi -i 0 --query-gpu=timestamp,memory.used,memory.free --format=csv --loop-ms=250` aufzeichnen. Den GPU-Index anpassen, die Abfrage direkt vor dem Lauf starten und danach beenden. Das Maximum ist ein abgetasteter, geräteweiter Wert. Ohne andere GPU-Prozesse lässt sich die Veränderung gegenüber dem Ausgangswert dem Versuch zuordnen. Maximalen Prozess-RSS aus `time -v` übernehmen und bei CPU-Offload zusätzlich Swap beobachten. Kontext-/Compute-Buffer, CPU-Puffer und Offload aus den Backend-Logs übernehmen. Ein leeres GPU-Messfeld bedeutet fehlende Messung.

Das Tool speichert die native Modell-Ladezeit und die Zeit der frischen Vektorberechnung separat. Das Performance-Profil misst für jeden Pilotprompt einen zusätzlichen ungestörten Prefill ohne Sampling. TTFT der eigentlichen Antworten enthält die Schritte bis zum ersten Ausgabetoken. Tokens/s bezeichnet die Dekodierphase ab dem ersten Token, mit offenem Wert bei zu kurzen Antworten. Prefill, TTFT, Tokens/s und Antwortlänge separat auswerten. Ladezeit hängt auch vom Dateisystem-Cache ab. Reihenfolge und Warm-/Kaltzustand dokumentieren und für Geschwindigkeitsvergleiche mehrere Durchläufe vorsehen.

Für jede geplante Variante einen Status festhalten, mit Quant, Profil, Seed, Modellhash, GPU-Argument, tatsächlichem Offload, Status, Run-ID und Logpfad. `unsupported` für nachgewiesene Format- oder Brückenprobleme, `OOM` für belegten Speichermangel und `failed` für andere Fehler verwenden. Ein ausstehender Versuch bleibt `pending`. Wenn für eine Variante der gemeinsame Offload wegen OOM gesenkt wird, die Referenz unter denselben Bedingungen erneut laufen lassen oder den Vergleich als nicht hinreichend kontrolliert markieren. Keine bestehenden Ergebnisse überschreiben.

## Auswertung

Antworten unter zufälligen Review-IDs mit Prompt und Lösungshinweis zur menschlichen Bewertung vorlegen. Die Zuordnung zu Quant, Layer und Alpha getrennt halten und erst nach der Bewertung in die Run-Historie zurückführen. Die vier vorhandenen Rubriken getrennt erfassen. Bei Safety Risk bewertet die Rubrik die Veränderung gegenüber der jeweiligen Baseline. Dieser harmlose Fragenkatalog misst allgemeine Modellsicherheit nur begrenzt.

```bash
python -m activation_steering.quant_study compare \
  runs/REFERENZ_RUN runs/QUANT_RUN > vergleich.json
```

`compare` paart vollständige Runs nach identischer Prompt-ID, Layer und Alpha. Es prüft unter anderem die Daten- und Template-Hashes, Einstellungen, Backend-Dateien und einen Tokenizer-Fingerprint über die tatsächlich verwendeten Konstruktions- und Evaluationstexte. Das Qualitätsprofil verlangt dieselbe Offload-Einstellung. Ob das Backend in beiden Fällen tatsächlich dieselben Layer auslagerte, anhand der Logs prüfen. Die Ausgabe enthält Roh-Normen, Kosinus-Ähnlichkeit pro Layer und pro Antwort die menschlichen Bewertungen, Baseline-Differenzen und Laufzeitmessungen. Fehlende Bewertungen und Kosinuswerte bei Nullnorm bleiben offen.

Für jeden Layer und Alpha zunächst die pro Prompt gepaarten Baseline-Differenzen auswerten. Dann die Differenzen zwischen Quant und Referenz vergleichen, jeweils mit Anzahl verfügbarer Paare und Streuung. Baselinequalität separat vergleichen. Geschwindigkeit und Speicher anhand des Performance-Profils auswerten. Bei nicht deterministischen Wiederholungen die geplante Temperatur und die Seeds vor dem ersten Run festlegen, dieselbe Konfigurationsdatei für alle Quants verwenden und pro Seed gegeneinander auswerten. `--settings` kann eine eigene, zuvor versionierte Pilotdatei laden und `--seed` einen festen Seed je Run setzen. Bei Greedy-Sampling erzeugen zusätzliche Seed-Werte keine unabhängigen Stichproben. Für Ausreißer und Ausfälle die Run- und Prompt-IDs im Bericht angeben.

Der spätere Bericht enthält die Matrix einschließlich Fehlschlägen, Baseline- und Steering-Qualität, Vektorvergleich sowie getrennte Ressourcen- und Laufzeitmessungen. Der gesamte GPU-Teil bleibt bis zur Ausführung auf der Tesla T4 offen.
