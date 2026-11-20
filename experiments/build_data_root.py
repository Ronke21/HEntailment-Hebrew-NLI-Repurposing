#!/usr/bin/env python3
"""
Build data/experiments/, the DATA_ROOT the experiment configs point at.

  - heq / qasem / hesum / hebsummaries / heb_paraphrase: copied from
    data/corpus/<src>/. train.jsonl is a class-balanced sample of PER_LABEL
    rows per label (seeded; 3,000 rows per source); dev and test are full.
  - hebnli / lchaim / winograd and their subsets (hebnli_15k, lchaim_max,
    hebnli_100 ... hebnli_5k, aug_hebnli_*): linked from data/stage1.
  - hent_all: the pooled train/dev of the five sources (seeded shuffle).
Idempotent: existing outputs are kept (delete a directory to rebuild it).
"""
from __future__ import annotations
import json, random, shutil
from collections import defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
CORPUS = REPO / "data/corpus"
V3D = REPO / "data/stage1"
OUT = REPO / "data/experiments"
SRC = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
LINK = ["hebnli", "lchaim", "winograd", "hebnli_15k", "lchaim_max",
        "hebnli_100", "hebnli_500", "hebnli_1k", "hebnli_5k",
        "aug_hebnli_100", "aug_hebnli_500", "aug_hebnli_1k", "aug_hebnli_5k", "aug_hebnli_15k"]
PER_LABEL = 1000
SEED = 42


def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def dump(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_source(src: str) -> bool:
    d = OUT / src
    if (d / "train.jsonl").exists():
        print(f"  {src}: exists, kept"); return True
    if not (CORPUS / src / "gate.json").exists():
        print(f"  {src}: corpus not finished (no gate.json) -- skipped"); return False
    train = load(CORPUS / src / "train.jsonl")
    by_label = defaultdict(list)
    for r in train: by_label[r["label"]].append(r)
    cap = min(PER_LABEL, min(len(v) for v in by_label.values()))
    rows = []
    for lab in ("entailment", "contradiction", "neutral"):
        v = by_label[lab]; random.Random(SEED).shuffle(v); rows += v[:cap]
    random.Random(SEED + 1).shuffle(rows)
    dump(d / "train.jsonl", rows)
    for sp in ("dev", "test"):
        shutil.copy2(CORPUS / src / f"{sp}.jsonl", d / f"{sp}.jsonl")
    print(f"  {src}: train {len(rows)} rows ({cap}/label of {dict((k, len(v)) for k, v in by_label.items())}); "
          f"dev {sum(1 for _ in open(d/'dev.jsonl'))} test {sum(1 for _ in open(d/'test.jsonl'))}")
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ready = [s for s in SRC if build_source(s)]
    for name in LINK:
        link = OUT / name; target = V3D / name
        if not target.exists(): print(f"  {name}: missing in data/stage1, skipped"); continue
        if link.is_symlink() or link.exists(): continue
        link.symlink_to(target); print(f"  linked {name} -> data/stage1")
    if set(ready) == set(SRC) and not (OUT / "hent_all" / "train.jsonl").exists():
        train, dev = [], []
        for s in SRC:
            train += load(OUT / s / "train.jsonl"); dev += load(OUT / s / "dev.jsonl")
        random.Random(SEED).shuffle(train); random.Random(SEED).shuffle(dev)
        dump(OUT / "hent_all/train.jsonl", train); dump(OUT / "hent_all/dev.jsonl", dev)
        print(f"  hent_all: train {len(train)} dev {len(dev)}")
    elif set(ready) != set(SRC):
        print(f"  hent_all: waiting on {sorted(set(SRC) - set(ready))}")
    print("DONE" if set(ready) == set(SRC) else "PARTIAL")


if __name__ == "__main__":
    main()
