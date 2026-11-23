"""Generate the fixed speaker-identity pilot from the shared ten prompts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from activation_steering.prompts import PROMPT_DIR, SHARED_SYCOPHANCY, load_prompt_set
from activation_steering.storage import atomic_bytes


STUDY_PATH = PROMPT_DIR.parent / "identity_bias" / "study.toml"
SET_NAME = "sycophancy_identity_pilot"
SCHEMA = 2


def build_prompts(study, prompt_set):
    fictional_name = study["fictional_name"]
    speakers = {
        "anonymous": "",
        **study["public_names"],
        "fictional": fictional_name,
        "fictional_title": f"{fictional_name} ({study['fictional_title']})",
    }
    if len(set(speakers.values())) != len(speakers):
        raise ValueError("Jeder Sprecher braucht einen eigenen Vorspann.")

    prompts = []
    for case in tomllib.loads(prompt_set.source)["prompts"]:
        for speaker_id, name in speakers.items():
            prefix = f"Fiktive Zuschreibung: {name}. " if name else ""
            prompts.append({
                **case,
                "id": f"ib-{case['id']}__{speaker_id}",
                "prompt": prefix + case["prompt"],
                "claim_id": case["id"],
                "claim_text": case["prompt"],
                "speaker_id": speaker_id,
            })
    seed = study["order_seed"]
    return sorted(prompts, key=lambda row: sha256(f"{seed}:{row['id']}".encode()).digest())


def write_prompt_set(directory):
    source = STUDY_PATH.read_bytes()
    study = tomllib.loads(source.decode("utf-8"))
    shared = load_prompt_set(SHARED_SYCOPHANCY)
    header = {
        "name": SET_NAME,
        "description": "Die zehn gemeinsamen Fragen mit je sieben Sprechervarianten.",
        "identity_study_schema": SCHEMA,
        "study_sha256": sha256(source).hexdigest(),
        "base_prompt_set": shared.name,
        "base_prompt_set_sha256": shared.hash,
        "order_seed": study["order_seed"],
    }

    def fields(values):
        return "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items())

    prompts = build_prompts(study, shared)
    body = "\n\n".join("[[prompts]]\n" + fields(prompt) for prompt in prompts)
    path = directory / f"{SET_NAME}.toml"
    atomic_bytes(path, (fields(header) + "\n\n" + body + "\n").encode("utf-8"))
    return path, len(prompts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", choices=("pilot",), help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=PROMPT_DIR)
    args = parser.parse_args()
    path, count = write_prompt_set(args.output_dir)
    print(f"{path}: {count} Prompts")


if __name__ == "__main__":
    main()
