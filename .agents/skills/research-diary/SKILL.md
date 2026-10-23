---
name: research-diary
description: Persönliches Forschungs- und Lerntagebuch pflegen (diary/YYYY-MM-DD.md). Nur auf ausdrückliche Anweisung Einträge erstellen oder ergänzen, sonst nur vorschlagen.
---

# Skill: research-diary

## Zweck

Das Forschungstagebuch unter `diary/` ist das persönliche Lern- und Forschungstagebuch des
Entwicklers. Es ist kein automatisches Commit-Log.

Dort stehen zum Beispiel:

- was heute gelernt wurde
- gelesene Artikel, Papers, Quellen
- angesehene Videos
- neue Ideen
- getroffene Forschungsentscheidungen
- durchgeführte Experimente
- auffällige Ergebnisse
- was nicht funktioniert hat
- offene Fragen
- geänderte Annahmen
- überlegte nächste Schritte

## Grundregel

Einträge nur erstellen oder ergänzen, wenn der Entwickler es ausdrücklich verlangt
(z. B. `Tagebuch`, `research-diary`, `Trage das ins Tagebuch ein`).

Nicht ungefragt ständig Einträge erzeugen.

Bei passenden Arbeitsschritten darf der Agent einmal kurz fragen, zum Beispiel:

- `Möchtest du, dass ich die heutige Erkenntnis im Forschungstagebuch dokumentiere?`
- `Soll ich die heutigen Tests und Ergebnisse kurz im Tagebuch festhalten?`

## Ablage und Format

- Ordner: `diary/`
- Eine Datei pro Tag: `diary/YYYY-MM-DD.md`
- Existiert die Tagesdatei bereits, sinnvoll ergänzen statt überschreiben.
- Einfache, gut lesbare Struktur, zum Beispiel:

```markdown
# 2026-10-06

## Was ich gemacht habe

## Was ich gelernt habe

## Quellen

## Erkenntnisse / Ideen

## Offene Fragen

## Nächste mögliche Schritte
```

## Regeln

- Nur Abschnitte verwenden, für die es Inhalt gibt. Keine leeren Abschnitte erzwingen.
- Menschlich und praktisch schreiben, keine automatisch wirkende Unternehmensdokumentation.
- Keine erfundenen Informationen. Theorien, Ideen und unbelegte Beobachtungen klar als
  solche kennzeichnen, nicht als bestätigte Experimentergebnisse darstellen.
- Quellen mit Titel, Autor, Plattform und Datum angeben. URLs nur eintragen, wenn sie
  verlässlich vorliegen. Keine URLs erfinden.
- Technische Projektänderungen gehören in `CHANGELOG.md`, nicht ins Tagebuch.
  Das Tagebuch hält Lernverlauf und Interpretation fest.
