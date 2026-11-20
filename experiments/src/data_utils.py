#!/usr/bin/env python3
"""
data_utils.py — Loading and mixing the repurposed NLI datasets.

All splits come from Data_Repurposing/repurposed_data/<dataset>/{train,dev,test}.jsonl
with fields: id, premise, hypothesis, label (entailment/contradiction/neutral),
dataset, split.
"""

from __future__ import annotations

import json
import pathlib
import random

import config


def load_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(dataset: str, split: str) -> list[dict]:
    """Load one split of one dataset, keeping only examples with a valid label."""
    path = config.DATA_ROOT / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run the Data_Repurposing conversion scripts first "
            "(see Data_Repurposing/README.md)."
        )
    data = load_jsonl(path)
    kept = [ex for ex in data
            if ex.get("label") in config.LABEL2ID
            and ex.get("premise") and ex.get("hypothesis")]
    if len(kept) < len(data):
        print(f"[data] {dataset}/{split}: dropped {len(data) - len(kept)} "
              f"examples with invalid labels or empty premise/hypothesis")
    if config.SMOKE_MODE:
        cap = config.SMOKE_DATA_CAPS.get(split)
        if cap and len(kept) > cap:
            kept = kept[:cap]
            print(f"[data] SMOKE: {dataset}/{split} capped to {cap}")
    # make sure downstream code can always rely on these fields
    for i, ex in enumerate(kept):
        ex.setdefault("id", f"{dataset}_{split}_{i:06d}")
        ex.setdefault("dataset", dataset)
    return kept


def load_train_dev(train_spec: str, seed: int,
                   max_per_dataset: int | None = None) -> tuple[list[dict], list[dict]]:
    """Return (train, dev) for a train spec.

    train_spec is one of config.TRAIN_DATASETS, config.ALL_TASKS (plain
    concatenation of the six datasets) or config.ALL_BALANCED (each dataset's
    train contribution capped at config.ALL_BALANCED_CAP so heq/qasem don't
    drown out winograd). max_per_dataset overrides the cap for either mixture.
    """
    rng = random.Random(seed)

    if train_spec not in (config.ALL_TASKS, config.ALL_BALANCED):
        return load_split(train_spec, "train"), load_split(train_spec, "dev")

    cap = max_per_dataset
    if train_spec == config.ALL_BALANCED and cap is None:
        cap = config.ALL_BALANCED_CAP

    train: list[dict] = []
    dev:   list[dict] = []
    for ds in config.TRAIN_DATASETS:
        ds_train = load_split(ds, "train")
        if cap is not None and len(ds_train) > cap:
            rng.shuffle(ds_train)
            ds_train = ds_train[:cap]
            print(f"[data] {train_spec} mixture: capped {ds} train to {cap}")
        train.extend(ds_train)
        dev.extend(load_split(ds, "dev"))
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def load_all_test_sets() -> dict[str, list[dict]]:
    """Return {dataset_name: test examples} for all 8 evaluation test sets."""
    return {ds: load_split(ds, "test") for ds in config.EVAL_DATASETS}


def load_cached_predictions(pred_file: pathlib.Path,
                            examples: list[dict]) -> list[str] | None:
    """Eval-resume: if a complete predictions file from a previous (crashed)
    run exists and matches *examples* by id and count, return its predictions
    so the test set is not re-scored. Returns None if absent or inconsistent."""
    if not pred_file.exists():
        return None
    try:
        rows = load_jsonl(pred_file)
    except Exception:
        return None
    if len(rows) != len(examples):
        return None
    if any(r.get("id") != ex["id"] for r, ex in zip(rows, examples)):
        return None
    return [r["prediction"] for r in rows]
