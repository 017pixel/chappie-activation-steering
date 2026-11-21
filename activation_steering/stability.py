"""Prepare the fixed Sycophancy vector-stability study without a model."""

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
from pathlib import Path
import re
import tomllib
import unicodedata

import tomlkit

from .config import ROOT
from .datasets import load_dataset
from .evaluation import RUBRIC_VERSION
from .prompts import load_prompt_set
from .storage import atomic_json


POOL = ROOT / "data/stability/sycophancy_stability_pool.toml"
HOLDOUT = "sycophancy_stability_holdout"
TOPICS = ("naturwissenschaft", "geschichte_kultur")
FRAMES = ("", "Kurz gesagt: ", "Zur Einordnung: ", "Die Antwort lautet: ")
SEEDS = (17, 43, 101, 227, 419)
SIZES = (5, 10, 20, 40)
THEME_SIZE = 12
MODEL = "Qwen3.8-27B-IQ4_XS.gguf"
QUANTIZATION = "IQ4_XS"
LAYERS = [8, 32, 56]
ALPHAS = [-1.0, 1.0, 2.0]
GENERATION = {
    "seed": 42, "temperature": 0.0, "top_k": 40, "top_p": 1.0, "min_p": 0.0,
    "repeat_penalty": 1.1, "max_output_tokens": 512, "enable_thinking": False,
}
CONTEXT_SIZE = 8192
TOKEN_POSITION = "last_non_padding"
NORMALIZATION = "none"
PROTOCOL_VERSION = 1
PRESSURES = ("direkt", "autorität", "peer", "subtil")


def check_leakage(train, test):
    """Check claim IDs and wording, including corrections; semantic review is separate."""
    def normalized(text):
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))

    entries = [(split, claim_id, [normalized(text) for text in texts])
               for split, claims in (("Train", train), ("Holdout", test))
               for claim_id, texts in claims]
    for (split_a, id_a, texts_a), (split_b, id_b, texts_b) in combinations(entries, 2):
        if id_a == id_b:
            raise ValueError(f"Doppelte Claim-ID: {split_a}/{id_a} und {split_b}/{id_b}.")
        for a in texts_a:
            for b in texts_b:
                matcher = SequenceMatcher(None, a, b, autojunk=False)
                if matcher.quick_ratio() >= 0.85 and matcher.ratio() >= 0.85:
                    raise ValueError(f"Ähnliche Aussagen: {split_a}/{id_a} und {split_b}/{id_b}.")


def study_inputs():
    pool = load_dataset(POOL)
    holdout = load_prompt_set(HOLDOUT)
    pairs = tomllib.loads(pool.source)["pairs"]
    test = tomllib.loads(holdout.source)["prompts"]

    if Counter((pair.get("topic"), pair.get("frame")) for pair in pairs) != {
        (topic, frame): 6 for topic in TOPICS for frame in range(1, len(FRAMES) + 1)
    }:
        raise ValueError("Der Pool braucht sechs Paare je Thema und Satzrahmen.")
    if Counter((prompt["category"], prompt["pressure"]) for prompt in holdout.prompts) != {
        (topic, pressure): 1 for topic in (*TOPICS, "alltag_technik") for pressure in PRESSURES
    }:
        raise ValueError("Der Holdout braucht je Thema einen Prompt pro Drucktyp.")

    train_claims = []
    for pair in pairs:
        claim_id = pair.get("id")
        if not isinstance(claim_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", claim_id):
            raise ValueError(f"Ungültige Claim-ID: {claim_id}.")
        prefix = FRAMES[pair["frame"] - 1]
        texts = [pair["target"], pair["contrast"]]
        if any(not text.startswith(prefix) or not text.endswith(".") or "\n" in text for text in texts):
            raise ValueError(f"Abweichender Satzrahmen: {claim_id}.")
        if not prefix and any(text.startswith(frame) for text in texts for frame in FRAMES[1:]):
            raise ValueError(f"Unerwarteter Satzrahmen: {claim_id}.")
        target, contrast = [text.removeprefix(prefix) for text in texts]
        if target == contrast or abs(len(target.split()) - len(contrast.split())) > 4:
            raise ValueError(f"Ungeeignetes Target-/Contrast-Paar: {claim_id}.")
        train_claims.append((claim_id, [target, contrast]))
    test_claims = []
    for row in test:
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"Falschaussage im Holdout fehlt: {row['id']}.")
        test_claims.append((row["id"], [claim, row["prompt"], row["correct_answer"]]))
    check_leakage(train_claims, test_claims)
    return pool, pairs, holdout


def select_pairs(pairs, group, seed, size):
    """Rank stable claim IDs with SHA-256; each condition has its own draw."""
    def rank(pair):
        return hashlib.sha256(f"v{PROTOCOL_VERSION}/{group}/{size}/{seed}/{pair['id']}".encode()).digest()

    if group == "size":
        selected = sorted(pairs, key=rank)[:size]
    elif group in TOPICS:
        candidates = [pair for pair in pairs if pair["topic"] == group]
        selected = []
        for frame in range(1, len(FRAMES) + 1):
            matching = (pair for pair in candidates if pair["frame"] == frame)
            selected.extend(sorted(matching, key=rank)[:size // len(FRAMES)])
    elif group == "reference":
        selected = pairs
    else:
        raise ValueError(f"Unbekannte Gruppe: {group}.")
    return sorted(selected, key=lambda pair: pair["id"])


def prepare(output: Path):
    pool, pairs, holdout = study_inputs()
    conditions = [("size", size, seed) for size in SIZES for seed in SEEDS]
    conditions += [(topic, THEME_SIZE, seed) for topic in TOPICS for seed in SEEDS]
    conditions.append(("reference", len(pairs), None))

    planned = []
    seen = set()
    for group, size, seed in conditions:
        key = f"{group}_{size}" if group == "size" else group
        name = f"stability_{key}" if seed is None else f"stability_{key}_s{seed}"
        selected = select_pairs(pairs, group, seed, size)
        claim_ids = tuple(pair["id"] for pair in selected)
        if len(claim_ids) != size or claim_ids in seen:
            raise ValueError(f"Falsche Paarzahl oder identische Ziehung: {name}.")
        seen.add(claim_ids)
        document = {
            "name": name, "description": f"Stabilitätsstudie, Gruppe {group}, {size} Paare, Seed {seed}.",
            "target_label": pool.target_label, "contrast_label": pool.contrast_label,
            "research_use": True, "pool_hash": pool.hash, "group": group, "size": size,
        }
        if seed is not None:
            document["selection_seed"] = seed
        document["pairs"] = selected
        content = tomlkit.dumps(document).encode("utf-8")
        planned.append(({
            "name": name, "group": group, "size": size, "seed": seed,
            "claim_ids": claim_ids, "topics": dict(Counter(pair["topic"] for pair in selected)),
            "dataset_hash": hashlib.sha256(content).hexdigest(),
        }, content))

    output.mkdir()
    for condition, content in planned:
        path = output / f"{condition['name']}.toml"
        path.write_bytes(content)
        load_dataset(condition["name"], output)
    (output / holdout.path.name).write_bytes(holdout.source.encode("utf-8"))
    # Write last so an incomplete materialization cannot be used for analysis.
    atomic_json(output / "manifest.json", {
        "protocol_version": PROTOCOL_VERSION, "pool_hash": pool.hash, "rubric_version": RUBRIC_VERSION,
        "prompt_set_name": holdout.name, "prompt_set_hash": holdout.hash,
        "prompt_ids": [prompt["id"] for prompt in holdout.prompts],
        "model_file": MODEL, "quantization": QUANTIZATION, "layers": LAYERS, "alphas": ALPHAS,
        "generation": GENERATION, "context_size": CONTEXT_SIZE,
        "token_position": TOKEN_POSITION, "normalization": NORMALIZATION,
        "conditions": [condition for condition, _ in planned],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Neues Verzeichnis für Datensätze und Manifest.")
    args = parser.parse_args()
    try:
        prepare(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Datensätze und Manifest in {args.output}")


if __name__ == "__main__":
    main()
