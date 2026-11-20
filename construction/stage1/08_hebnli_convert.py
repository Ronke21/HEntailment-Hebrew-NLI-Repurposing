"""
Task 8 — HebNLI Conversion
Hebrew translation of MNLI-style sentence pairs (NNLP-IL/HebNLI).
Direct mapping: translation1→premise, translation2→hypothesis, label→NLI label.
No Gemini, no heuristics, no embeddings — labels are gold.
Test split uses manually verified hebrew_label; train/val use original_label.
No length filter (labels are already gold NLI pairs). Skips splits already done.
"""

import argparse
import json
import random
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_utils import (setup_logging, is_split_done, log_label_stats, log_examples)

import logging
logger = logging.getLogger(__name__)

LABEL_NORM = {
    "entailment":    "entailment",
    "Entailment":    "entailment",
    "contradiction": "contradiction",
    "Contradiction": "contradiction",
    "neutral":       "neutral",
    "Neutral":       "neutral",
}

SPLIT_FILES = {
    "train": ("HebNLI_train.jsonl", "original_label"),
    "dev":   ("HebNLI_val.jsonl",   "original_label"),
    "test":  ("HebNLI_test.jsonl",  "hebrew_label"),
}


def convert_split(split: str, fpath: Path, label_field: str,
                  output_dir: Path, smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    records = [json.loads(l) for l in fpath.open(encoding="utf-8") if l.strip()]
    if smoke > 0:
        rng_smoke = random.Random(42)
        records = rng_smoke.sample(records, min(smoke, len(records)))
        logger.info(f"[{split}] SMOKE MODE: {len(records)} rows")
    logger.info(f"[{split}] Loaded {len(records)} records from {fpath.name} | label_field={label_field}")

    out_rows = []
    skipped = 0

    for i, rec in enumerate(records):
        raw_label = rec.get(label_field, "")
        label = LABEL_NORM.get(raw_label)
        if label is None:
            skipped += 1
            continue
        prem = rec.get("translation1", "").strip()
        hyp  = rec.get("translation2", "").strip()
        if not prem or not hyp:
            skipped += 1
            continue
        out_rows.append({
            "id":       f"hebnli_{split}_{i:06d}",
            "premise":  prem,
            "hypothesis": hyp,
            "label":    label,
            "genre":    rec.get("genre", ""),
            "pairID":   rec.get("pairID", ""),
            "dataset":  "hebnli",
            "split":    split,
        })

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "hebnli")
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | skipped={skipped} | {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="HebNLI → NLI JSONL")
    parser.add_argument("--data_dir",   required=True, help="Path to HebNLI/ directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: limit to N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_dir), "08_hebnli_convert")
    logger.info("HebNLI conversion started" + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, (fname, label_field) in SPLIT_FILES.items():
        total += convert_split(split, data_dir / fname, label_field,
                               output_dir, smoke=args.smoke)
    logger.info(f"HebNLI done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
