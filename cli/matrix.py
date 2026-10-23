"""Layer/Alpha-Matrix mit getrennten Baselines und zwei Achsen Pagination."""

from statistics import mean

from rich import box
from rich.table import Table
from rich.text import Text

from activation_steering.evaluation import METRICS, latest_evaluations
from activation_steering.runs import read_metadata, read_results, read_snapshot_config
from cli.common import console, say
from cli.review import browse_answers, show_rubric

PERFORMANCE = {
    "output_tokens": "Output Tokens", "ttft": "TTFT (s)",
    "tokens_per_second": "TPS (tok/s)", "generation_time": "Generation Time (s)",
}
MATRIX_METRICS = {**{key: value["label"] for key, value in METRICS.items()}, **PERFORMANCE}


def cell_value(rows, metric, evaluations, expected=1):
    """Gibt nur gemessene Werte aus, mit transparenter Teilbewertung."""
    if metric in METRICS:
        values = [evaluations.get((row["result_id"], metric), {}).get("score") for row in rows]
    else:
        values = [row.get(metric) for row in rows]
    values = [value for value in values if value is not None]
    total = max(expected, len(rows))
    if not values:
        label = "?" if metric in METRICS else "-"
        if total > 1:
            label += f"\n(0/{total})"
        return label, None
    value = mean(values)
    if metric == "steering_effect":
        label = f"{value:+.2g}"
    elif metric == "output_tokens":
        label = f"{value:.5g}"
    else:
        label = f"{value:.3g}"
    if total > 1 or len(values) != total:
        label += f"\n({len(values)}/{total})"
    return label, value


def cell_style(value, metric):
    if value is None:
        return "dim"
    if metric == "steering_effect":
        return "cyan" if value < 0 else "bold blue" if value > 0 else "white"
    if metric in METRICS:
        maximum = METRICS[metric]["max"]
        return "dim cyan" if value < maximum / 3 else "cyan" if value < maximum * 2 / 3 else "bold cyan"
    return None


def page_size(width, height, *, multi=False):
    """Eine weitere Alpha-Seite erhält Werte, die auf schmalen Terminals fehlen."""
    return max(1, (height - 18) // (2 if multi else 1)), max(1, (width - 10) // 14)


def render_matrix(run_dir, metric, row_page=0, alpha_page=0):
    metadata = read_metadata(run_dir)
    config = read_snapshot_config(run_dir)
    resolved = config.get("resolved", {})
    layers = resolved.get("layers", metadata.get("layers", []))
    alphas = resolved.get("alphas", metadata.get("alphas", []))
    expected = len(resolved.get("prompts", [config.get("prompt", "")]))
    results = read_results(run_dir)
    evaluations = latest_evaluations(run_dir)
    rows_per_page, alphas_per_page = page_size(console.size.width, console.size.height, multi=expected > 1)
    row_pages = max(1, (len(layers) + rows_per_page - 1) // rows_per_page)
    alpha_pages = max(1, (len(alphas) + alphas_per_page - 1) // alphas_per_page)
    row_page = min(max(row_page, 0), row_pages - 1)
    alpha_page = min(max(alpha_page, 0), alpha_pages - 1)
    selected_layers = layers[row_page * rows_per_page:(row_page + 1) * rows_per_page]
    selected_alphas = alphas[alpha_page * alphas_per_page:(alpha_page + 1) * alphas_per_page]
    say(f"\nMatrix: {run_dir.name} | {MATRIX_METRICS[metric]}", style="bold")
    baseline = [row for row in results if row["type"] == "baseline"]
    baseline_label = "nicht anwendbar" if metric == "steering_effect" else cell_value(baseline, metric, evaluations, expected)[0]
    say(f"Gemeinsame Baseline ({expected} Prompt{'s' if expected != 1 else ''}): {baseline_label}")
    if not alphas:
        say("Dieser Lauf enthält ausschließlich Baselines.")
    else:
        table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False)
        table.add_column("Layer", width=5, overflow="fold")
        for alpha in selected_alphas:
            table.add_column(f"{alpha:+g}", width=11, overflow="fold")
        for layer in selected_layers:
            cells = [Text(str(layer))]
            for alpha in selected_alphas:
                matches = [row for row in results if row["type"] == "steering"
                           and row.get("layer") == layer and row.get("alpha") == alpha]
                label, value = cell_value(matches, metric, evaluations, expected)
                cells.append(Text(label, style=cell_style(value, metric)))
            table.add_row(*cells)
        console.print(table)
    say(f"Layerseite {row_page + 1}/{row_pages} | Alpha-Seite {alpha_page + 1}/{alpha_pages}")
    if selected_layers:
        say(f"Layer: {selected_layers} | Alpha: {selected_alphas}")
    if metric in METRICS:
        show_rubric(metric)
        say("? = noch nicht reviewed. (n/total) = bewertete Antworten / geplante Antworten.")
    else:
        say("- = nicht gemessen. (n/total) = verfügbare Werte / geplante Antworten.")
    say("Farben markieren Werte und Richtungen. Sie sind keine Gut/Schlecht-Bewertung.")
    return row_page, alpha_page, row_pages, alpha_pages


def matrix_view(run_dir):
    metric = "refusal"
    row_page = alpha_page = 0
    while True:
        row_page, alpha_page, row_pages, alpha_pages = render_matrix(run_dir, metric, row_page, alpha_page)
        raw = input("n/p Layerseite | a/z Alpha-Seite | m Metrik | l Layer öffnen | q zurück: ").strip().lower()
        if raw == "q":
            return
        if raw == "n":
            row_page = min(row_pages - 1, row_page + 1)
        elif raw == "p":
            row_page = max(0, row_page - 1)
        elif raw == "a":
            alpha_page = min(alpha_pages - 1, alpha_page + 1)
        elif raw == "z":
            alpha_page = max(0, alpha_page - 1)
        elif raw == "m":
            keys = list(MATRIX_METRICS)
            for index, key in enumerate(keys, 1):
                say(f"[{index}] {MATRIX_METRICS[key]}")
            try:
                index = int(input("Metriknummer: ")) - 1
                if not 0 <= index < len(keys):
                    raise ValueError("Ungültige Metriknummer.")
                metric = keys[index]
            except ValueError as error:
                say(error)
        elif raw == "l":
            try:
                browse_answers(run_dir, layer=int(input("Layer öffnen: ")))
            except ValueError as error:
                say(error)
