#!/usr/bin/env python3
"""
Balanced vs as-built evaluation, from saved predictions (CPU, seconds).

The test sets ship with the label distribution the construction produces.
Every run writes results/<model>/<spec>/<seed>/predictions/<testset>.jsonl with
one {id, gold, prediction} per test row. This script recomputes macro-F1 (and
accuracy) on a seeded, label-balanced subsample of each test set (the same
subsample for every run) and writes both numbers side by side.

  python3 rescore_balanced.py --root experiments/grid            # all runs
  python3 rescore_balanced.py --root experiments/grid --seeds 3  # 3 subsamples, mean +- SD
Output: <root>/balanced_rescore.json and a printed table.
"""
from __future__ import annotations
import argparse, json, random, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("entailment", "contradiction", "neutral")


def macro_f1(pairs) -> float:
    f1s = []
    for lab in LABELS:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fn == 0:
            continue                      # label absent from this test set
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else float("nan")


def balanced_indices(golds, seed: int) -> list[int]:
    """Indices of a label-balanced subsample: n = the rarest label's count."""
    by = defaultdict(list)
    for i, g in enumerate(golds):
        by[g].append(i)
    n = min(len(v) for v in by.values())
    rng = random.Random(seed)
    keep = []
    for lab in sorted(by):
        idx = by[lab][:]; rng.shuffle(idx); keep += idx[:n]
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="experiment root with results/")
    ap.add_argument("--seeds", type=int, default=3, help="how many balanced subsamples to average")
    ap.add_argument("--min-per-label", type=int, default=30, help="skip a test set whose rarest label is smaller")
    args = ap.parse_args()
    root = Path(args.root)
    out = {}
    rows = []
    for pred in sorted(root.glob("results/*/*/*/predictions/*.jsonl")):
        model, spec, seed = pred.parts[-5], pred.parts[-4], pred.parts[-3]
        testset = pred.stem
        data = [json.loads(l) for l in open(pred, encoding="utf-8")]
        golds = [d["gold"] for d in data]; preds = [d["prediction"] for d in data]
        counts = Counter(golds)
        if len(counts) < 2 or min(counts.values()) < args.min_per_label:
            continue
        full = macro_f1(list(zip(golds, preds)))
        bal = []
        for s in range(args.seeds):
            idx = balanced_indices(golds, 42 + s)
            bal.append(macro_f1([(golds[i], preds[i]) for i in idx]))
        key = f"{model}/{spec}/{seed}/{testset}"
        out[key] = {"n": len(data), "label_counts": dict(counts),
                    "macro_f1_full": round(full, 4),
                    "macro_f1_balanced_mean": round(st.mean(bal), 4),
                    "macro_f1_balanced_sd": round(st.stdev(bal), 4) if len(bal) > 1 else 0.0,
                    "delta": round(st.mean(bal) - full, 4)}
        rows.append((testset, spec, full, st.mean(bal)))
    json.dump(out, open(root / "balanced_rescore.json", "w"), indent=1)
    agg = defaultdict(list)
    for testset, spec, full, bal in rows:
        agg[testset].append(bal - full)
    print(f"{'test set':18s} {'units':>6s} {'mean(balanced - full)':>22s} {'max |delta|':>12s}")
    for t, v in sorted(agg.items()):
        print(f"{t:18s} {len(v):6d} {st.mean(v):22.4f} {max(abs(x) for x in v):12.4f}")
    print(f"\n{len(out)} (unit, test set) pairs -> {root/'balanced_rescore.json'}")


if __name__ == "__main__":
    main()
