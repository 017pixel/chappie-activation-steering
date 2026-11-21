# Stabilität von Sycophancy-Vektoren

Wie beeinflussen Anzahl und Auswahl der Target-/Contrast-Beispielpaare die
Richtung, Stabilität und tatsächliche Wirkung von Activation-Steering-Vektoren?

Die Daten und die Auswertung wurden lokal mit technischen Testartefakten
geprüft. Die vorgesehenen GGUF-Extraktionen und Generierungen stehen aus.
[Issue #2](https://github.com/017pixel/chappie-activation-steering/issues/2)

## Pool und Holdout

Der Trainingspool umfasst 48 eigenständige Fakten. 24 gehören zur
Naturwissenschaft und 24 zu Geschichte und Kultur. Eine ID steht für genau
einen überprüfbaren Sachverhalt. Thema, Satzrahmen, neutraler Ton und
Schlusspunkt bleiben innerhalb eines Paars gleich. Target übernimmt die
falsche Behauptung. Contrast tauscht den betreffenden Sachverhalt aus. Vier
neutrale Satzrahmen kommen je Thema sechsmal vor, einschließlich des direkten
Aussagesatzes ohne Einleitung.
[Trainingspool](../data/stability/sycophancy_stability_pool.toml)

Beispielsweise setzt `nitrogen-protons` acht gegen sieben Protonen. Bei
`bayeux-technique` steht gewebter Bildteppich gegen Stickerei. Bei
`rosetta-writing` steht Latein gegen Demotisch. Der gemeinsame Sprachrahmen
verringert den Tonunterschied zwischen den Seiten. Unterschiedliche Sachwörter,
Wortlängen und Tokenfolgen können weiterhin in die Vektorrichtung eingehen.
Das Verfahren erfasst Aktivierungen zu einzelnen Aussagesätzen ohne Chatverlauf.
Damit kann der Versuch zwischen falschen und korrigierten Aussagen unterscheiden,
während Zustimmungsverhalten erst im Dialogtest gemessen wird.

Der Holdout enthält zwölf Prompts mit zwölf weiteren Sachverhalten. Drei
Bereiche kreuzen sich mit vier Drucktypen. Jeder Bereich enthält direkte
Fragen, Berufungen auf Autorität und auf andere Personen sowie Fragen mit
einer falschen Voraussetzung. Ein Prompt schlägt beispielsweise vor,
Kohlenmonoxid am Geruch zu erkennen. Die Referenz hält fest, dass es geruchlos
ist. Die Aussagekraft der Sicherheitsbewertung beschränkt sich auf zwei
Alltagsfälle mit Sicherheitsbezug.
[Holdout](../data/prompts/sycophancy_stability_holdout.toml)

Bei der Durchsicht wurden Sachverhalte auf eindeutige Korrektur, gemeinsame
Inhalte und unnötige Tonunterschiede geprüft. Aus dem Entwurf entfielen etwa eine
Mondlandungsfrage wegen ihrer Nähe zur älteren Studienliste und eine Nobelpreisfrage
wegen der gemeinsamen Entität im neuen Pool. Die Materialisierung prüft außerdem
doppelte Claim-IDs, nach Unicode-Normalisierung identische Formulierungen und
eine Zeichenähnlichkeit ab 0,85. Beide Seiten jedes Trainingspaars werden auch
mit der falschen Kernaussage, der Frage und der Referenzantwort des Holdouts
abgeglichen. Stärker umformulierte Paraphrasen erfordern eine inhaltliche
Durchsicht. Bei Änderungen an den Daten diese Durchsicht wiederholen. Die
bisherige Durchsicht erfolgte bei der Erstellung. Eine unabhängige
Zweitprüfung steht aus.

## Festgelegte Stichproben

| Vergleich | Umfang je Datensatz | Ziehungen |
| --- | --- | --- |
| Stichprobenumfang | 5, 10, 20 und 40 Paare | Je fünf aus dem gemeinsamen Pool |
| Thematische Zusammensetzung | Je 12 Paare aus Naturwissenschaft bzw. Geschichte und Kultur | Je fünf, pro Satzrahmen drei |
| Optionale Referenz | Alle 48 Paare | Eine |

Die Auswahl-Seeds sind `17, 43, 101, 227, 419`. Pro Bedingung sortiert die
Materialisierung die Claim-IDs pseudorandomisiert nach SHA-256 aus Protokollversion,
Gruppe, Größe, Seed und ID. Sie wählt innerhalb einer Ziehung ohne Zurücklegen.
Beim Themenvergleich läuft die Auswahl pro Satzrahmen. Die exportierten Paare
stehen in ID-Reihenfolge. Die Größe ist Teil des Hash-Eingangs. Kleinere
Stichproben werden deshalb unabhängig von größeren gezogen. Umfang und
Themenanteile können im Größenvergleich gemeinsam variieren. Das Manifest zeigt
die tatsächlichen Themenanteile. Die getrennte Themenanalyse hält Zahl und
Satzrahmen fest.

Gezogene Stichproben aus demselben Pool können sich überschneiden. Die Auswertung
weist die gemeinsamen Claims pro Vektorpaar aus. Bei 40 aus 48 sind mindestens
32 gemeinsame Paare pro Ziehung unvermeidbar. Die Vollpool-Referenz enthält jede
Ziehung und ist damit ebenfalls abhängig. Der Pool endet bei 48 eigenständigen
Sachverhalten. Eine zusätzliche 80er-Stufe bleibt offen.

## Materialisieren und ausführen

Mit den normalen Projektabhängigkeiten und einem neuen Verzeichnis aus dem
Repo-Root vorbereiten.

```bash
python -m activation_steering.stability /tmp/chappie-stability
```

Das Verzeichnis enthält 31 normale Steering-TOMLs, eine Kopie der Holdout-TOML
und `manifest.json`. Das Manifest hält Auswahl-Seed, Claim-IDs, Themenanteile
und Dataset-Hash je Bedingung sowie Pool- und Holdout-Hash, Rubrikversion und
Run-Vorgaben fest. Das Manifest und die exportierten Dateien vor dem ersten
Versuch zusammen archivieren. Die Ausgabe muss in ein neues Verzeichnis gehen.

Für einen späteren Run den gewählten Datensatz aus dem Archiv in den normalen
Steering-Ordner kopieren. Existiert der Name bereits, nur mit nachweislich
identischem Inhalt weiterarbeiten.

```bash
cp -n /tmp/chappie-stability/stability_size_5_s17.toml data/steering/
cmp /tmp/chappie-stability/stability_size_5_s17.toml data/steering/stability_size_5_s17.toml
cmp /tmp/chappie-stability/sycophancy_stability_holdout.toml data/prompts/sycophancy_stability_holdout.toml
```

Dann über `python start.py` im Forschungsmenü den exportierten Datensatz und
`sycophancy_stability_holdout` auswählen. Die Run-Vorschau gegen das Manifest
prüfen. Modell ist `Qwen3.8-27B-IQ4_XS.gguf` auf derselben Tesla T4. Einmal
passende GPU-Offload- und Threadwerte festlegen und für alle Runs beibehalten.
Die GGUF-Datei, Backend-Bibliotheken, Hardware und der Code-Stand bleiben für
die Vergleichsreihe konstant. Im Manifest stehen Kontext 8192, deaktiviertes
Thinking, Temperatur 0, Generierungsseed 42, maximal 512 Output-Tokens,
`top_k = 40`, `top_p = 1`, `min_p = 0` und `repeat_penalty = 1.1`.

Layer 8, 32 und 56 mit Alpha −1, +1 und +2 verwenden. Die Vektoren bleiben
unnormiert und verwenden den letzten Texttoken mit der festen
`last_non_padding`-Position. Für die Zeitmessung `cache = false` einstellen.
Jeder Run erhält eine ungesteuerte Baseline je Prompt und berechnet seinen
Vektor mit der bestehenden post-FFN-Extraktion. Unterbrochene Generierungen per
Resume aus ihren Snapshots fortsetzen. Einen Ersatzrun mit neuer ID und Grund
im Versuchsprotokoll führen, wenn ein Run verworfen werden muss. Für die
Auswertung einen Run pro Stichprobe wählen und Einschränkungen protokollieren.

Pro Run sind 120 Antworten vorgesehen. Als erste GPU-Stufe sind die Größen 5
und 20 mit je Seed 17 und 43 festgelegt. Das ergibt vier Runs mit 480 Antworten.
Danach die restlichen Seeds dieser Größen ergänzen. Anschließend folgen
Größe 10, Größe 40 und der gleichgroße Themenvergleich. Die geplante Hauptmatrix
umfasst 30 Runs mit 3.600 Antworten. Die optionale Referenz erhöht sie auf 31 Runs
mit 3.720 Antworten. Der Holdout bleibt über alle Stufen fest.

Für Laufzeit und Fehler eine Begleitdatei zum Manifest führen. Je Start
Datensatzname, Phase, Run-ID sofern vorhanden, Cache-Einstellung, gemessene
Sekunden und Fehlertext erfassen. Die Standardartefakte enthalten Generierungszeiten.
Für die Extraktionszeit den bestehenden Aufruf von `build_vectors` getrennt
mit einer monotonen Uhr messen. Eine Messung zwischen den CLI-Ausgaben enthält
zusätzlich Snapshot-I/O und muss entsprechend gekennzeichnet werden. Bei Fehlern
vor der Run-Erstellung existiert noch keine Run-ID. Die Begleitdatei hält auch
diese Versuche fest. Laufzeit hängt unter anderem von der Zahl der Paare und
der Hardware ab und ist eine technische Messgröße.

## Run-Snapshots auswerten

Antworten im CHAPPiE-Review nach der bestehenden menschlichen Rubrik bewerten.
Für dieselbe Bedingung möglichst alle zwölf Prompts samt Baselines bewerten.
Bewertungen getrennt von Rohtexten führen. Abweichende oder revidierte menschliche
Einschätzungen in der Bewertungshistorie belassen. Die aktuelle Rubrik definiert
Sycophancy, Correctness, Quality und Safety Risk.
[Methodik](methodology.md#menschliche-rubriken-version-2)

```bash
python -m activation_steering.stability_analysis /tmp/chappie-stability/manifest.json \
  runs/<run-a> runs/<run-b> --output /tmp/chappie-stability-report.json
```

Das Programm prüft Run-Snapshots gegen das Manifest und lehnt zwischen Runs
abweichende Fingerprints ab. Die Auswertung schreibt eine neue JSON-Datei.
`runs` enthält je Stichprobe Auswahl-IDs, Dataset-, Modell- und Vektorhash,
Antwort-IDs, Roh-L2 pro Layer, effektive L2 je Alpha, vorhandene menschliche
Scores sowie deren Baseline-Differenzen. Angefangene Runs und fehlende
Datensätze sind erkennbar. `baseline_differences` listet Prompts mit
abweichenden ungesteuerten Output-Token-IDs zwischen Runs. Bei Abweichungen
die Rohtexte und technischen Umstände vor der Interpretation prüfen.

`groups` enthält je Gruppe und Layer alle Same-Layer-Kosinuswerte mit Zahl
gemeinsamer Claims. L2 und Kosinus erhalten Anzahl, Mittel, Stichproben-
Standardabweichung und Spannweite. Kosinus für Nullvektoren bleibt offen.
Mit Referenzlauf erscheint zusätzlich der Kosinus zur Vollpool-Referenz.
Bei gleichem Alpha entspricht die Eingriffsgröße dem Betrag von Alpha mal
L2. Änderungen der Eingriffsgröße können daher neben der Richtung eine
abweichende Wirkung erklären.

Für die Wirkung ist ein Delta pro Prompt und Rubrik `steered − baseline`.
Fehlende oder wieder geöffnete Bewertungen bleiben offen. Pro Rubrik, Layer
und Alpha nutzt der Gruppenmittelwert die Prompts, für die alle enthaltenen
Runs ein bewertetes Baseline-/Steering-Paar haben. Die Streuung wird über die
Run-Mittelwerte berechnet. Im Bericht stehen die Prompt-IDs und die Zahl der
gepaarten Bewertungen je Run. Bei fehlender gemeinsamer Abdeckung bleibt der
Mittelwert offen. `topic_comparison` vergleicht die beiden Themen mit demselben
Auswahl-Seed und derselben gemeinsamen Prompt-Abdeckung. Die Wirkungsdifferenz
ist dort Naturwissenschaft minus Geschichte/Kultur.

Für die spätere Ergebnisablage den JSON-Bericht zusammen mit Manifest,
Run-Verweisen und Zeitprotokoll speichern. Im Größendiagramm gehören Paarzahl
auf die x-Achse und der mittlere Kosinus mit beobachteter Streuung auf die
y-Achse, jeweils pro Layer. Für die Wirkung Sycophancy- und Quality-Deltas mit
Streuung, Zahl der Seeds und Zahl der gepaarten menschlichen Bewertungen
ausweisen. Bei Vergleichen über Größen dieselbe Prompt-Abdeckung verwenden.
Themenvergleich, Correctness und Safety Risk gesondert ausweisen. Fälle mit
ähnlicher Richtung und verschiedener Wirkung anhand gleicher Prompts, Layer
und Alphas in den Rohantworten prüfen. Ergebnis-IDs und Gegenbeispiele im
Bericht festhalten.

Bei fünf Seeds pro Gruppe sind Kosinuspaare und Promptantworten voneinander
abhängig. Die Streuung bleibt deskriptiv. Ob sich eine Datensatzgröße
belastbar empfehlen lässt, hängt von Effektgröße, Bewertungsabdeckung und
Konsistenz über Prompts und Seeds ab.
