"""
Task 4 — HeSum Conversion
V2 only: (full article, journalist summary) E/C/N.
Contradiction by sentence-aware heuristics. NeoDictaBERT neutral.
Skips splits whose output JSONL already exists.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.embed_utils import encode, select_neutral_pairs
from utils.contradiction_utils import apply_heuristics_sentence_aware
from utils.logging_utils import (setup_logging, is_split_done, log_gpu_info,
                                  log_label_stats, log_examples, log_heuristic_breakdown,
                                  length_ok, log_length_rejections)

import logging
logger = logging.getLogger(__name__)

# Only salient heuristics survive one-sentence hypothesis vs. full-article premise
_HESUM_ALLOWED = {
    "verb_negation", "existential_negation",
    "number_substitution", "number_substitution_heb",
    "quantifier_flip", "superlative_flip",
}

SPLIT_FILES = {
    "train": "train/train.csv",
    "dev":   "dev/validation.csv",
    "test":  "test/test.csv",
}


def convert_split(split: str, df: pd.DataFrame, output_dir: Path, random_state: int,
                  smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    if smoke > 0:
        df = df.sample(min(smoke, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"[{split}] SMOKE MODE: random sample of {len(df)} rows")
    logger.info(f"[{split}] {len(df)} articles | V2 only (article→summary)")

    heuristic_rng = random.Random(random_state)
    heuristic_counts: dict = {}

    # Embeddings + neutral
    t1 = time.time()
    articles  = df["article"].tolist()
    summaries = df["summary"].tolist()
    emb = encode(articles, cache_path=output_dir / f"embeddings_{split}.npy")
    neutral_pairs = select_neutral_pairs(emb, list(range(len(df))), random_state=random_state)
    logger.info(f"[{split}] Embeddings + neutral done | {time.time()-t1:.1f}s")

    out_rows = []
    rejected: dict[str, int] = {}

    for i, (_, row) in enumerate(df.iterrows()):
        article = row["article"]
        summary = row["summary"]
        base    = {"dataset": "hesum", "split": split}

        if length_ok(article, summary):
            out_rows.append({**base, "id": f"hesum_{split}_{i:05d}_E_v2",
                             "premise": article, "hypothesis": summary,
                             "label": "entailment", "version": "v2"})
        else:
            rejected["entailment"] = rejected.get("entailment", 0) + 1

        modified, heuristic = apply_heuristics_sentence_aware(
            summary, rng=heuristic_rng, allowed=_HESUM_ALLOWED,
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified:
            if length_ok(article, modified):
                heuristic_counts[heuristic] = heuristic_counts.get(heuristic, 0) + 1
                out_rows.append({**base, "id": f"hesum_{split}_{i:05d}_C_v2",
                                 "premise": article, "hypothesis": modified,
                                 "label": "contradiction", "version": "v2",
                                 "contradiction_heuristic": heuristic})
            else:
                rejected["contradiction"] = rejected.get("contradiction", 0) + 1

    for i, j in neutral_pairs:
        base = {"dataset": "hesum", "split": split}
        if length_ok(articles[i], summaries[j]):
            out_rows.append({**base, "id": f"hesum_{split}_{i:05d}_N_v2",
                             "premise": articles[i], "hypothesis": summaries[j],
                             "label": "neutral", "version": "v2"})
        else:
            rejected["neutral"] = rejected.get("neutral", 0) + 1
    log_length_rejections(logger, rejected, split)

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "hesum")
    log_heuristic_breakdown(logger, out_rows, split)
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | total: {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="HeSum → NLI JSONL")
    parser.add_argument("--data_dir",     required=True, help="Path to HeSum/data/")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: limit to N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(Path(args.log_dir), "02_hesum_convert")
    log_gpu_info(logger)
    logger.info("HeSum conversion started" + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, rel in SPLIT_FILES.items():
        df = pd.read_csv(data_dir / rel)
        total += convert_split(split, df, output_dir, args.random_state, smoke=args.smoke)
    logger.info(f"HeSum done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
