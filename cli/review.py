"""Menschliche Bewertung und unveränderte Rohantworten im Terminal."""

from activation_steering.evaluation import METRICS, evaluate, latest_evaluations, summarize
from activation_steering.runs import read_results
from activation_steering.storage import read_json
from cli.common import menu, performance, say


def baseline_for(result, results):
    return next((row for row in results if row["type"] == "baseline"
                 and row.get("prompt_id") == result.get("prompt_id")
                 and row.get("repeat_id", 0) == result.get("repeat_id", 0)), None)


def applicable_metrics(result):
    return [metric for metric in METRICS
            if metric != "steering_effect" or result["type"] != "baseline"]


def show_result(run_dir, result, *, results=None, evaluations=None):
    results = read_results(run_dir) if results is None else results
    evaluations = latest_evaluations(run_dir) if evaluations is None else evaluations
    say(f"\nResult-ID: {result['result_id']}", style="bold")
    say("Baseline" if result["type"] == "baseline" else
        f"Layer: {result['layer']} | Alpha: {result['alpha']:+g}")
    say(f"Prompt-ID: {result.get('prompt_id', 'nicht verfügbar')}")
    say(f"\nPrompt:\n{result['prompt']}")
    baseline = baseline_for(result, results)
    if result["type"] != "baseline":
        steering = read_json(run_dir / "steering_metadata.json")
        say(f"\nTarget: {steering.get('target_label', '?')} (positives Alpha)")
        say(f"Contrast: {steering.get('contrast_label', '?')} (negatives Alpha)")
        say(steering.get("dataset_description", ""))
        say("\nBaseline:\n" + (baseline["response"] if baseline else "Noch nicht generiert."))
    say(f"\n{'Baseline-Antwort' if result['type'] == 'baseline' else 'Gesteuerte Antwort'}:\n{result['response']}")
    say("\nPerformance")
    performance(result)
    say("\nMenschliche Bewertung")
    for metric in applicable_metrics(result):
        latest = evaluations.get((result["result_id"], metric), {})
        score = latest.get("score")
        status = f"{score} (bewertet: {latest.get('reviewed_at', '?')})" if score is not None else "? (offen)"
        say(f"{METRICS[metric]['label']}: {status}")


def show_rubric(metric):
    rubric = METRICS[metric]
    say(f"\n{rubric['label']}", style="bold")
    say(rubric["description"])
    for score, description in rubric["scores"].items():
        say(f"{score}: {description}")


def _pick_metric():
    keys = list(METRICS)
    for index, metric in enumerate(keys, 1):
        say(f"[{index}] {METRICS[metric]['label']}")
    raw = input("Metriknummer, q zurück: ").strip()
    if raw.lower() == "q":
        return None
    try:
        index = int(raw) - 1
        if 0 <= index < len(keys):
            return keys[index]
    except ValueError:
        pass
    raise ValueError("Ungültige Metriknummer.")


def review_candidates(results, evaluations, metric=None, layer=None, alpha=None, reviewed=False):
    candidates = []
    for row in results:
        if layer is not None and row.get("layer") != layer:
            continue
        if alpha is not None and row.get("alpha") != alpha:
            continue
        keys = [metric] if metric else applicable_metrics(row)
        keys = [key for key in keys if key in applicable_metrics(row)]
        if not keys:
            continue
        scores = [evaluations.get((row["result_id"], key), {}).get("score") for key in keys]
        if (reviewed and any(score is not None for score in scores)) or (
                not reviewed and any(score is None for score in scores)):
            candidates.append(row)
    return candidates


def review_answers(run_dir, *, metric=None, layer=None, alpha=None, reviewed=False):
    results = read_results(run_dir)
    evaluations = latest_evaluations(run_dir)
    candidates = review_candidates(results, evaluations, metric, layer, alpha, reviewed)
    if not candidates:
        say("Keine passenden Antworten. Alle gewählten Bewertungen erledigt oder noch keine Rohantworten.")
        return
    for index, result in enumerate(candidates, 1):
        say(f"\nAntwort {index} / {len(candidates)}")
        show_result(run_dir, result, results=results, evaluations=evaluations)
        keys = [metric] if metric else applicable_metrics(result)
        if not reviewed:
            keys = [key for key in keys if evaluations.get((result["result_id"], key), {}).get("score") is None]
        for key in keys:
            show_rubric(key)
            while True:
                raw = input("Score, Enter beibehalten, - wieder offen, s überspringen, q zurück: ").strip().lower()
                if raw == "q":
                    return
                if raw == "s":
                    break
                if not raw:
                    break
                try:
                    score = None if raw == "-" else int(raw)
                    evaluate(run_dir, result["result_id"], key, score)
                    evaluations = latest_evaluations(run_dir)
                    say("Bewertung gespeichert." if score is not None else "Bewertung wieder offen.")
                    break
                except (ValueError, OSError) as error:
                    say(f"Bewertung nicht gespeichert: {error}")
            if raw == "s":
                break


def review_inbox(run_dir):
    while True:
        summary = summarize(run_dir)
        say(f"\nReview Inbox: {run_dir.name}", style="bold")
        say(f"Antworten: {summary['completed']} / {summary['total']} generiert")
        for metric, counts in summary["review_counts"].items():
            say(f"{METRICS[metric]['label']}: {counts['reviewed']} / {counts['total']}")
        choice = menu("Review", [("1", "Nächste offene Antworten"), ("2", "Bestimmte Metrik"),
                                  ("3", "Bestimmten Layer"), ("4", "Bestimmtes Alpha"),
                                  ("5", "Bereits bewertete Antworten"), ("6", "Zurück")])
        try:
            if choice == "1":
                review_answers(run_dir)
            elif choice == "2":
                metric = _pick_metric()
                if metric:
                    review_answers(run_dir, metric=metric)
            elif choice == "3":
                review_answers(run_dir, layer=int(input("Layer: ")))
            elif choice == "4":
                review_answers(run_dir, alpha=float(input("Alpha (0 = Baseline): ")))
            elif choice == "5":
                review_answers(run_dir, reviewed=True)
            elif choice in ("6", "q"):
                return
        except (ValueError, OSError) as error:
            say(f"Review nicht möglich: {error}")


def browse_answers(run_dir, *, layer=None):
    results = [row for row in read_results(run_dir) if layer is None or row.get("layer") == layer]
    if not results:
        say("Noch keine passenden Antworten gespeichert.")
        return
    page = 0
    while True:
        per_page = 10
        start = page * per_page
        for index, row in enumerate(results[start:start + per_page], start + 1):
            label = "Baseline" if row["type"] == "baseline" else f"Layer {row['layer']}, Alpha {row['alpha']:+g}"
            say(f"[{index}] {label} | {row['result_id']}")
        raw = input("Antwortnummer öffnen, n nächste, p vorige, q zurück: ").strip().lower()
        if raw == "q":
            return
        if raw == "n":
            page = min((len(results) - 1) // per_page, page + 1)
        elif raw == "p":
            page = max(0, page - 1)
        else:
            try:
                index = int(raw) - 1
                if not 0 <= index < len(results):
                    raise ValueError("Ungültige Antwortnummer.")
                show_result(run_dir, results[index], results=results)
                input("Enter zur Antwortliste: ")
            except ValueError as error:
                say(error)
