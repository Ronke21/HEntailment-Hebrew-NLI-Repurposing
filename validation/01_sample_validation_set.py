#!/usr/bin/env python3
"""
Stratified sampler for the HEntailment human-validation study.

Draws a random, seeded sample from the released corpus
(data/stage1), stratified by source dataset x
gold label, and prepares:

  1. validation_set_master.{jsonl,csv}  - full sample with source dataset,
     repurposing method, premise, hypothesis, gold label. For the study
     team only; never shown to annotators.
  2. annotator_1.csv / annotator_2.csv  - blind files (premise + hypothesis
     only, shuffled, no source/method/gold label) for the two human
     annotators to fill in a predicted class.
  3. blind_key.csv                      - item_id -> gold label / dataset /
     repurposing method, to score annotators and compute IAA later. Keep
     this out of the annotators' hands.
  4. sampling_stats.csv + printed report - premise/hypothesis word-length
     mean/std for the sample vs. the full corpus population, per source and
     per label, to check the sample isn't skewed relative to HEntailment as
     a whole.

Usage:
    conda activate heb_nli_mrl_eval
    python 01_sample_validation_set.py
    python 01_sample_validation_set.py --seed 123 --n-per-class 21
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

SOURCES = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
LABELS = ["entailment", "contradiction", "neutral"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# Per-dataset, per-label construction method, following
# Data_Repurposing/README.md's "NLI Label Sources" table. Contradiction rows
# carry their own `contradiction_heuristic` field, which is used verbatim
# instead of this generic description when present.
METHOD_BY_SOURCE_LABEL = {
    "heq": {
        "entailment": "QA gold answer -> Gemini declarative composition",
        "contradiction": "QA wrong-answer -> Gemini declarative composition",
        "neutral": "NeoDictaBERT cosine-distance far pair",
    },
    "qasem": {
        "entailment": "QASem gold answer -> Gemini declarative composition",
        "contradiction": "Cross-sentence wrong-answer substitution",
        "neutral": "NeoDictaBERT cosine-distance far pair",
    },
    "hesum": {
        "entailment": "Article/title -> summary (same-article)",
        "contradiction": "Rule-based heuristic on summary",
        "neutral": "NeoDictaBERT cosine-distance far pair",
    },
    "hebsummaries": {
        "entailment": "Article/title -> summary (same-document)",
        "contradiction": "Rule-based heuristic on summary",
        "neutral": "NeoDictaBERT cosine-distance far pair",
    },
    "heb_paraphrase": {
        "entailment": "Original <-> paraphrase (bidirectional)",
        "contradiction": "Rule-based heuristic on paraphrase",
        "neutral": "NeoDictaBERT cosine-distance far pair",
    },
}


def word_len(text):
    return len(text.split())


def repurposing_method(dataset, label, row):
    if label == "contradiction" and row.get("contradiction_heuristic"):
        return f"heuristic:{row['contradiction_heuristic']}"
    return METHOD_BY_SOURCE_LABEL[dataset][label]


def load_split(corpus_dir, dataset, split):
    path = corpus_dir / dataset / f"{split}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def population_stats(rows_by_source):
    """Mean/std premise & hypothesis word length, per source x label, over
    the full corpus split (not just the sample)."""
    stats = {}
    for dataset, rows in rows_by_source.items():
        for label in LABELS:
            sub = [r for r in rows if r["label"] == label]
            p_lens = np.array([word_len(r["premise"]) for r in sub])
            h_lens = np.array([word_len(r["hypothesis"]) for r in sub])
            stats[(dataset, label)] = {
                "n": len(sub),
                "premise_mean": p_lens.mean() if len(sub) else float("nan"),
                "premise_std": p_lens.std(ddof=1) if len(sub) > 1 else float("nan"),
                "hypothesis_mean": h_lens.mean() if len(sub) else float("nan"),
                "hypothesis_std": h_lens.std(ddof=1) if len(sub) > 1 else float("nan"),
            }
    return stats


def sample_stratum(rows, label, n, rng):
    pool = [r for r in rows if r["label"] == label]
    if len(pool) < n:
        raise ValueError(
            f"Not enough {label} examples to sample {n} (pool has {len(pool)})"
        )
    # sort by id first so the sample is reproducible regardless of file
    # read order, then draw with the seeded RNG
    pool = sorted(pool, key=lambda r: r.get("id", ""))
    return rng.sample(pool, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                     help="Path to the released corpus (repurposed_data_v2b)")
    ap.add_argument("--split", default="test",
                     help="Which split to sample from (default: test, the benchmark's own eval split)")
    ap.add_argument("--sources", nargs="+", default=SOURCES)
    ap.add_argument("--n-per-class", type=int, default=21,
                     help="Examples per (source, label) stratum. 21 -> 63/source, 315 total over 5 sources.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "sampled_data")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading '{args.split}' split for {len(args.sources)} sources from {args.corpus_dir} ...")
    rows_by_source = {ds: load_split(args.corpus_dir, ds, args.split) for ds in args.sources}
    for ds, rows in rows_by_source.items():
        counts = {lab: sum(1 for r in rows if r["label"] == lab) for lab in LABELS}
        print(f"  {ds}: {len(rows)} rows  {counts}")

    # --- draw the stratified sample ---
    sample = []
    for ds in args.sources:
        rows = rows_by_source[ds]
        for label in LABELS:
            picked = sample_stratum(rows, label, args.n_per_class, rng)
            for r in picked:
                sample.append({
                    "dataset": ds,
                    "split": args.split,
                    "orig_id": r.get("id", ""),
                    "label": label,
                    "repurposing_method": repurposing_method(ds, label, r),
                    "premise": r["premise"],
                    "hypothesis": r["hypothesis"],
                    "premise_len_words": word_len(r["premise"]),
                    "hypothesis_len_words": word_len(r["hypothesis"]),
                })

    # stable, content-derived item_id so re-running with the same seed
    # reproduces identical ids
    for i, item in enumerate(sample):
        item["item_id"] = f"hval_{i:04d}"

    total = len(sample)
    n_sources = len(args.sources)
    per_source = args.n_per_class * len(LABELS)
    print(f"\nSampled {total} items ({per_source}/source x {n_sources} sources, "
          f"{args.n_per_class}/class), seed={args.seed}")

    # --- 1. master file (has everything, study-team only) ---
    master_jsonl = args.out_dir / "validation_set_master.jsonl"
    master_csv = args.out_dir / "validation_set_master.csv"
    with open(master_jsonl, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    master_fields = ["item_id", "dataset", "split", "orig_id", "label", "repurposing_method",
                      "premise", "hypothesis", "premise_len_words", "hypothesis_len_words"]
    with open(master_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=master_fields)
        w.writeheader()
        w.writerows(sample)
    print(f"Wrote master file: {master_jsonl.name}, {master_csv.name}")

    # --- 2. blind key (gold labels, kept by the study team) ---
    blind_key_fields = ["item_id", "dataset", "label", "repurposing_method", "orig_id"]
    with open(args.out_dir / "blind_key.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind_key_fields)
        w.writeheader()
        w.writerows({k: item[k] for k in blind_key_fields} for item in sample)
    print("Wrote blind_key.csv (do NOT share with annotators)")

    # --- 3. blind annotation files (no dataset/method/label) ---
    blind_order = sample[:]
    rng.shuffle(blind_order)  # single shared shuffle so both annotators see items in the same, non-block order
    blind_fields = ["item_id", "premise", "hypothesis", "annotator_label"]
    for ann in (1, 2):
        path = args.out_dir / f"annotator_{ann}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=blind_fields)
            w.writeheader()
            for item in blind_order:
                w.writerow({
                    "item_id": item["item_id"],
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "annotator_label": "",  # fill with: entailment / contradiction / neutral
                })
        print(f"Wrote {path.name} (blind, {len(blind_order)} items, fill 'annotator_label')")

    # --- 4. representativeness stats: sample vs. full-corpus population ---
    pop_stats = population_stats(rows_by_source)
    stats_rows = []
    for ds in args.sources:
        for label in LABELS:
            sub = [it for it in sample if it["dataset"] == ds and it["label"] == label]
            p_lens = np.array([it["premise_len_words"] for it in sub])
            h_lens = np.array([it["hypothesis_len_words"] for it in sub])
            pop = pop_stats[(ds, label)]
            stats_rows.append({
                "dataset": ds,
                "label": label,
                "n_sample": len(sub),
                "n_population": pop["n"],
                "sample_premise_mean": round(p_lens.mean(), 2),
                "population_premise_mean": round(pop["premise_mean"], 2),
                "sample_premise_std": round(p_lens.std(ddof=1), 2) if len(sub) > 1 else float("nan"),
                "population_premise_std": round(pop["premise_std"], 2),
                "sample_hypothesis_mean": round(h_lens.mean(), 2),
                "population_hypothesis_mean": round(pop["hypothesis_mean"], 2),
                "sample_hypothesis_std": round(h_lens.std(ddof=1), 2) if len(sub) > 1 else float("nan"),
                "population_hypothesis_std": round(pop["hypothesis_std"], 2),
            })

    stats_path = args.out_dir / "sampling_stats.csv"
    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"Wrote {stats_path.name}\n")

    # printed report
    print(f"{'dataset':<16}{'label':<14}{'n':>4}  "
          f"{'premise len (sample/pop)':<34}{'hypothesis len (sample/pop)':<28}")
    for row in stats_rows:
        p = f"{row['sample_premise_mean']:.1f}+-{row['sample_premise_std']:.1f} / {row['population_premise_mean']:.1f}+-{row['population_premise_std']:.1f}"
        h = f"{row['sample_hypothesis_mean']:.1f}+-{row['sample_hypothesis_std']:.1f} / {row['population_hypothesis_mean']:.1f}+-{row['population_hypothesis_std']:.1f}"
        print(f"{row['dataset']:<16}{row['label']:<14}{row['n_sample']:>4}  {p:<34}{h:<28}")

    with open(args.out_dir / "sampling_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "split": args.split,
            "corpus_dir": str(args.corpus_dir),
            "sources": args.sources,
            "n_per_class": args.n_per_class,
            "total_items": total,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSeed and run parameters saved to sampling_manifest.json (seed={args.seed})")


if __name__ == "__main__":
    main()
