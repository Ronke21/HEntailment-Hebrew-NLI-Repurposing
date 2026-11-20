"""
Task 1 — LCHAIM Conversion
Direct 1:1 label mapping of the DIKTA variant. No Gemini, no GPU.
Skips splits whose output JSONL already exists and is non-empty.
"""

import argparse
import json
import random
import time
from pathlib import Path

from utils.logging_utils import (setup_logging, is_split_done, log_label_stats, log_examples)

import logging
logger = logging.getLogger(__name__)

LABEL_MAP = {"e": "entailment", "c": "contradiction", "n": "neutral"}

SPLIT_FILES = {
    "train": "dikta_heb_train.json",
    "dev":   "dikta_heb_dev.json",
    "test":  "dikta_heb_test.json",
}


def convert_split(data_dir: Path, split: str, filename: str, output_dir: Path,
                  smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    fpath = data_dir / filename
    records = json.loads(fpath.read_text(encoding="utf-8"))
    if smoke > 0:
        rng_smoke = random.Random(42)
        records = rng_smoke.sample(records, min(smoke, len(records)))
        logger.info(f"[{split}] SMOKE MODE: random sample of {len(records)} records")
    logger.info(f"[{split}] Loaded {len(records)} records from {fpath.name}")

    rows = []
    skipped = 0
    for i, rec in enumerate(records):
        label = LABEL_MAP.get(rec.get("label", ""))
        if label is None:
            skipped += 1
            continue
        rows.append({
            "id": f"lchaim_{split}_{i:05d}",
            "premise":    rec["premise"],
            "hypothesis": rec["hypothesis"],
            "label":      label,
            "uid":        rec.get("uid", ""),
            "dataset":    "lchaim",
            "split":      split,
        })

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    log_label_stats(logger, rows, split, "lchaim")
    log_examples(logger, rows, split)
    logger.info(f"[{split}] Wrote {len(rows):,} rows → {out_path} | skipped:{skipped} | {elapsed:.1f}s")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert LCHAIM DIKTA dataset to NLI JSONL.")
    parser.add_argument("--data_dir",   required=True, help="Path to lchaim_dikta_dataset/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: limit to N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(Path(args.log_dir), "04_lchaim_convert")
    logger.info(f"LCHAIM conversion started | data={data_dir} output={output_dir}"
                + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, filename in SPLIT_FILES.items():
        total += convert_split(data_dir, split, filename, output_dir, smoke=args.smoke)
    logger.info(f"LCHAIM done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
