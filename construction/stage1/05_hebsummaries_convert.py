"""
Task 6 — HebSummaries Conversion
V2 only: (text_raw, summary). V1 (metadata_title, summary) is excluded.
Length constraint: all emitted rows require len(premise) >= 1.2 × len(hypothesis).
Contradiction heuristics. NeoDictaBERT neutral.
Gold/silver label retained. Skips splits whose output JSONL already exists.
"""

import argparse
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.embed_utils import encode, select_neutral_pairs
from utils.contradiction_utils import apply_heuristics_sentence_aware
from utils.logging_utils import (setup_logging, is_split_done, log_gpu_info,
                                  log_label_stats, log_examples, log_heuristic_breakdown,
                                  length_ok, log_length_rejections)

import logging
logger = logging.getLogger(__name__)

SPLIT_FILES = {
    "train": "train.jsonl",
    "dev":   "val.jsonl",
    "test":  "test.jsonl",
}


def parse_metadata(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return {}


def load_split(fpath: Path) -> list[dict]:
    rows = []
    with fpath.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_split(split: str, records: list[dict], output_dir: Path, random_state: int,
                  smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    if smoke > 0:
        import random as _rnd
        _rnd.seed(42)
        records = _rnd.sample(records, min(smoke, len(records)))
        logger.info(f"[{split}] SMOKE MODE: random sample of {len(records)} records")
    logger.info(f"[{split}] {len(records)} records")

    import random as _rnd_mod
    heuristic_rng = _rnd_mod.Random(random_state)
    heuristic_counts: dict = {}

    texts     = [r["text_raw"] for r in records]
    summaries = [r["summary"]  for r in records]
    metas     = [parse_metadata(r.get("metadata", "{}")) for r in records]
    titles    = [m.get("title", "") or "" for m in metas]
    types_    = [m.get("type", "unknown") for m in metas]
    sources   = [m.get("source", "") for m in metas]
    genres    = [m.get("genre", "") or m.get("category", "") for m in metas]

    empty_titles = sum(1 for t in titles if not t.strip())
    logger.info(f"[{split}] Titles from metadata: {len(titles)-empty_titles}/{len(titles)} present")
    from collections import Counter
    logger.info(f"[{split}] Type breakdown: {dict(Counter(types_))}")

    # Embeddings + neutral
    t1 = time.time()
    emb = encode(texts, cache_path=output_dir / f"embeddings_{split}.npy")
    neutral_pairs = select_neutral_pairs(emb, list(range(len(records))), random_state=random_state)
    logger.info(f"[{split}] Embeddings + neutral done | {time.time()-t1:.1f}s")

    out_rows = []
    rejected: dict[str, int] = {}

    for i, rec in enumerate(records):
        article = texts[i]
        summary = summaries[i]
        base    = {"dataset": "hebsummaries", "split": split,
                   "type": types_[i], "source": sources[i], "genre": genres[i]}

        if length_ok(article, summary):
            out_rows.append({**base, "id": f"hebsum_{split}_{i:05d}_E_v2",
                             "premise": article, "hypothesis": summary,
                             "label": "entailment", "version": "v2"})
        else:
            rejected["entailment"] = rejected.get("entailment", 0) + 1

        modified, heuristic = apply_heuristics_sentence_aware(
            summary, rng=heuristic_rng,
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified:
            if length_ok(article, modified):
                heuristic_counts[heuristic] = heuristic_counts.get(heuristic, 0) + 1
                out_rows.append({**base, "id": f"hebsum_{split}_{i:05d}_C_v2",
                                 "premise": article, "hypothesis": modified,
                                 "label": "contradiction", "version": "v2",
                                 "contradiction_heuristic": heuristic})
            else:
                rejected["contradiction"] = rejected.get("contradiction", 0) + 1

    for i, j in neutral_pairs:
        base = {"dataset": "hebsummaries", "split": split,
                "type": types_[i], "source": sources[i], "genre": genres[i]}
        if length_ok(texts[i], summaries[j]):
            out_rows.append({**base, "id": f"hebsum_{split}_{i:05d}_N_v2",
                             "premise": texts[i], "hypothesis": summaries[j],
                             "label": "neutral", "version": "v2"})
        else:
            rejected["neutral"] = rejected.get("neutral", 0) + 1
    log_length_rejections(logger, rejected, split)

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "hebsummaries")
    log_heuristic_breakdown(logger, out_rows, split)
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="HebSummaries → NLI JSONL")
    parser.add_argument("--data_dir",     required=True, help="Path to HebSummaries/")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: limit to N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_dir), "05_hebsummaries_convert")
    log_gpu_info(logger)
    logger.info("HebSummaries conversion started" + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, fname in SPLIT_FILES.items():
        records = load_split(data_dir / fname)
        total  += convert_split(split, records, output_dir, args.random_state, smoke=args.smoke)
    logger.info(f"HebSummaries done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
