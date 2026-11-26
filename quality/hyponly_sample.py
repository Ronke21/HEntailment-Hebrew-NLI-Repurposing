#!/usr/bin/env python
"""
03_sample_hyponly_eval.py -- balanced evaluation samples for the hypothesis-only study.

Draws, from the TEST split of each dataset, a class-balanced random sample:

  heq, qasem, hesum, hebsummaries, heb_paraphrase   300 each (100 per label)
  hebnli, lchaim                                    300 each (100 per label)
  hentailment_all (pooled over the 5 sources)      1500     (500 per label)

Everything is drawn from `repurposed_data_v2b`, the released corpus, under a
fixed seed so the same sample is reproducible and identical across every
classifier arm. These are EVALUATION sets only: each hypothesis-only
classifier is trained on the corresponding dataset's own train split and
scored here, so all arms are compared on exactly the same items.

The pooled sample is drawn independently of the per-source samples (it is a
fresh 500-per-label draw over the union), so per-source and pooled numbers
are not simply averages of one another.

USAGE
    python "03_sample_hyponly_eval.py"                 # writes samples/
    python "03_sample_hyponly_eval.py" --seed 7        # redraw
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import os
CORPUS = Path(os.environ.get("HYPONLY_CORPUS", REPO_ROOT / "data" / "corpus"))

SOURCES = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
TRANSLATED = ["hebnli", "lchaim"]
LABELS = ["entailment", "contradiction", "neutral"]

# 236 per label -> 708 per dataset. 236 is LCHAIM's smallest test class, so it
# is the largest EQUAL balanced sample all three compared datasets can supply.
# Same rule that sets the 1,900-per-label training cap, applied to evaluation,
# so HEntailment / HebNLI / LCHAIM rest on identical amounts of evidence.
PER_LABEL_SINGLE = 236
PER_LABEL_POOLED = 236


def load(dataset, split):
    path = CORPUS / dataset / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def balanced_sample(rows, per_label, rng, tag):
    by = defaultdict(list)
    for r in rows:
        if r.get("label") in LABELS:
            by[r["label"]].append(r)
    out = []
    for lab in LABELS:
        pool = by[lab]
        if len(pool) < per_label:
            raise SystemExit(f"{tag}: only {len(pool)} {lab} available, need {per_label}")
        out.extend(rng.sample(pool, per_label))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "hyponly_samples")
    ap.add_argument("--pooled-only", action="store_true",
                    help="draw only the pooled hentailment_all sample (QASem's test split "
                         "has fewer than 236 contradictions, so per-source samples are skipped)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    manifest = []
    for ds in ([] if a.pooled_only else SOURCES + TRANSLATED):
        rows = load(ds, "test")
        samp = balanced_sample(rows, PER_LABEL_SINGLE, rng, ds)
        (a.out_dir / f"{ds}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in samp) + "\n", encoding="utf-8")
        manifest.append(dict(name=ds, n=len(samp), per_label=PER_LABEL_SINGLE,
                             test_pool=len(rows), kind="source" if ds in SOURCES else "translated"))
        print(f"  {ds:16s} {len(samp):5d} sampled from {len(rows):6d}  {dict(Counter(r['label'] for r in samp))}")

    # Pooled HEntailment is balanced on BOTH axes: exactly PER_LABEL_POOLED per
    # label, and within each label spread as evenly as possible over the five
    # sources (47 or 48 each), so no source dominates the corpus-level number.
    by_src = {ds: load(ds, "test") for ds in SOURCES}
    samp = []
    for lab in LABELS:
        quota = [PER_LABEL_POOLED // len(SOURCES)] * len(SOURCES)
        for i in range(PER_LABEL_POOLED % len(SOURCES)):
            quota[i] += 1
        for ds, q in zip(SOURCES, quota):
            pool = [r | {"source": ds} for r in by_src[ds] if r.get("label") == lab]
            if len(pool) < q:
                raise SystemExit(f"pooled: {ds}/{lab} has {len(pool)}, need {q}")
            samp.extend(rng.sample(pool, q))
    rng.shuffle(samp)
    (a.out_dir / "hentailment_all.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in samp) + "\n", encoding="utf-8")
    manifest.append(dict(name="hentailment_all", n=len(samp), per_label=PER_LABEL_POOLED,
                         test_pool=sum(len(v) for v in by_src.values()), kind="pooled"))
    print(f"  {'hentailment_all':16s} {len(samp):5d} sampled from "
          f"{sum(len(v) for v in by_src.values()):6d}"
          f"  by source: {dict(Counter(r['source'] for r in samp))}")

    (a.out_dir / "manifest.json").write_text(
        json.dumps(dict(seed=a.seed, corpus=str(CORPUS), datasets=manifest),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} samples to {a.out_dir}")


if __name__ == "__main__":
    main()
