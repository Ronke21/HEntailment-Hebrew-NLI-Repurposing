"""
Task 1 — HeQ Conversion
Full-context premise only (windowed variant removed).
OpenRouter reformulates (question, answer) → declarative hypothesis.
Contradiction via 12 heuristics (random selection) on the entailment hypothesis.
NeoDictaBERT selects neutral pairs by embedding distance.
Retains question_quality and is_gold metadata.
Skips splits whose output JSONL already exists.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.gemini_utils import batch_generate, HYPOTHESIS_PROMPT
from utils.embed_utils import encode, select_neutral_pairs
from utils.contradiction_utils import apply_heuristics
from utils.logging_utils import (setup_logging, is_split_done, log_gpu_info,
                                  log_label_stats, log_examples, log_heuristic_breakdown,
                                  length_ok, log_length_rejections)

import logging
logger = logging.getLogger(__name__)

SPLIT_FILES = {
    "train": "train v1.1.json",
    "dev":   "val v1.1.json",
    "test":  "test v1.1.json",
}


def parse_heq_json(fpath: Path) -> list[dict]:
    data = json.loads(fpath.read_text(encoding="utf-8"))
    rows = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible", True):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                rows.append({
                    "context":          context,
                    "question":         qa["question"],
                    "answer_text":      answers[0]["text"],
                    "answer_start":     answers[0].get("answer_start") or answers[0].get("answer.start", 0),
                    "wrong_answers":    qa.get("wrong_answers", []),
                    "question_quality": qa.get("question quality", "unchecked"),
                    "is_gold":          qa.get("question quality", "") == "gold",
                    "qid":              qa.get("id", ""),
                })
    return rows


def convert_split(split: str, fpath: Path, output_dir: Path,
                  random_state: int, smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    logger.info(f"[{split}] Parsing {fpath.name}")
    rows = parse_heq_json(fpath)
    if smoke > 0:
        rng_smoke = random.Random(42)
        rows = rng_smoke.sample(rows, min(smoke, len(rows)))
        logger.info(f"[{split}] SMOKE MODE: random sample of {len(rows)} rows")
    logger.info(f"[{split}] {len(rows):,} answerable QA pairs | parse: {time.time()-t0:.1f}s")

    heuristic_rng = random.Random(random_state)
    heuristic_counts: dict = {}

    # --- Gemini: entailment hypotheses ---
    t1 = time.time()
    ent_items = [{"question": r["question"], "answer": r["answer_text"]} for r in rows]
    ent_hyps  = batch_generate(ent_items, HYPOTHESIS_PROMPT,
                                output_dir / f"gemini_ent_cache_{split}.json")
    logger.info(f"[{split}] Entailment hypotheses done | {time.time()-t1:.1f}s")

    # --- Contradiction via heuristics on entailment hypotheses ---
    t2 = time.time()
    con_rows = []
    for i, r in enumerate(rows):
        hyp = ent_hyps[i]
        if not hyp:
            continue
        modified, heuristic = apply_heuristics(
            hyp, rng=heuristic_rng,
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified:
            heuristic_counts[heuristic] = heuristic_counts.get(heuristic, 0) + 1
            con_rows.append((i, modified, heuristic))
    logger.info(f"[{split}] Contradiction heuristics: {len(con_rows):,}/{len(rows):,} rows | {time.time()-t2:.1f}s")

    # --- Embeddings + neutral selection ---
    t3 = time.time()
    contexts = [r["context"] for r in rows]
    emb = encode(contexts, cache_path=output_dir / f"embeddings_{split}.npy")
    doc_ids = [r["qid"].rsplit("-", 1)[0] if "-" in r["qid"] else r["qid"] for r in rows]
    neutral_pairs = select_neutral_pairs(emb, doc_ids, random_state=random_state)
    logger.info(f"[{split}] Embeddings + neutral selection done | {time.time()-t3:.1f}s")

    # --- Build output rows ---
    out_rows = []
    rejected: dict[str, int] = {}

    for i, r in enumerate(rows):
        hyp = ent_hyps[i]
        if not hyp:
            continue
        if not length_ok(r["context"], hyp):
            rejected["entailment"] = rejected.get("entailment", 0) + 1
            continue
        base = {
            "premise":          r["context"],
            "question_quality": r["question_quality"],
            "is_gold":          r["is_gold"],
            "dataset":          "heq",
            "split":            split,
        }
        out_rows.append({**base, "id": f"heq_{split}_{i:05d}_E",
                         "hypothesis": hyp, "label": "entailment"})

    for i, modified_hyp, heuristic in con_rows:
        r = rows[i]
        if not length_ok(r["context"], modified_hyp):
            rejected["contradiction"] = rejected.get("contradiction", 0) + 1
            continue
        out_rows.append({
            "id":                      f"heq_{split}_{i:05d}_C",
            "premise":                  r["context"],
            "hypothesis":               modified_hyp,
            "label":                    "contradiction",
            "question_quality":         r["question_quality"],
            "is_gold":                  r["is_gold"],
            "contradiction_heuristic":  heuristic,
            "dataset":                  "heq",
            "split":                    split,
        })

    for i, j in neutral_pairs:
        hyp = ent_hyps[j]
        if not hyp:
            continue
        r = rows[i]
        if not length_ok(r["context"], hyp):
            rejected["neutral"] = rejected.get("neutral", 0) + 1
            continue
        out_rows.append({
            "id":               f"heq_{split}_{i:05d}_N",
            "premise":          r["context"],
            "hypothesis":       hyp,
            "label":            "neutral",
            "question_quality": r["question_quality"],
            "is_gold":          r["is_gold"],
            "dataset":          "heq",
            "split":            split,
        })
    log_length_rejections(logger, rejected, split)

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "heq")
    log_heuristic_breakdown(logger, out_rows, split)
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | total: {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="HeQ → NLI JSONL")
    parser.add_argument("--data_dir",     required=True, help='Path to "data v1.1" dir')
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: random N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(Path(args.log_dir), "01_heq_convert")
    log_gpu_info(logger)
    logger.info("HeQ conversion started" + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, fname in SPLIT_FILES.items():
        total += convert_split(split, data_dir / fname, output_dir,
                               args.random_state, smoke=args.smoke)
    logger.info(f"HeQ done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
