"""
Task 7 — Hebrew QASem Conversion
Subsample 10K rows from train. Gemini Flash de-segments morpheme-segmented Hebrew.
Full sentence as premise. Gemini entailment hypotheses.
Contradiction via heuristics A+C (verb_negation + entity_swap from premise).
NeoDictaBERT neutral. Skips splits whose output JSONL already exists.
"""

import argparse
import ast
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils.gemini_utils import batch_generate, HYPOTHESIS_PROMPT, DESEG_PROMPT
from utils.embed_utils import encode, select_neutral_pairs
from utils.contradiction_utils import apply_heuristics
from utils.logging_utils import (setup_logging, is_split_done, log_gpu_info,
                                  log_label_stats, log_examples,
                                  length_ok, log_length_rejections)

import logging
logger = logging.getLogger(__name__)

SPLIT_FILES = {
    "train": "He-train-dataset-QASem.csv",
    "dev":   "He-gold-dataset-QASem-dev.csv",
    "test":  "He-gold-dataset-QASem-test.csv",
}


def parse_answers(raw: str) -> list[str]:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return [raw] if raw else []


def convert_split(split: str, df: pd.DataFrame, output_dir: Path,
                  random_state: int, smoke: int = 0) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    df = df.copy()
    df["answers_list"] = df["answers"].apply(parse_answers)
    df["answer_text"]  = df["answers_list"].apply(lambda a: a[0] if a else "")
    df = df[df["answer_text"] != ""].reset_index(drop=True)

    if smoke > 0:
        rng_smoke = random.Random(42)
        idx = rng_smoke.sample(range(len(df)), min(smoke, len(df)))
        df = df.iloc[sorted(idx)].reset_index(drop=True)
        logger.info(f"[{split}] SMOKE MODE: random sample of {len(df)} rows")

    logger.info(f"[{split}] {len(df):,} rows after filtering empty answers")
    logger.info(f"[{split}] Predicate types: {df['type'].value_counts().to_dict()}")

    # Gemini desegmentation of morpheme-split sentences
    t1 = time.time()
    deseg_items = [{"sentence": row["sentence"]} for _, row in df.iterrows()]
    deseg_texts = batch_generate(deseg_items, DESEG_PROMPT,
                                  output_dir / f"gemini_deseg_cache_{split}.json")
    df["sentence_deseg"] = deseg_texts
    empty_deseg = sum(1 for t in deseg_texts if not t.strip())
    logger.info(f"[{split}] Deseg done: {len(deseg_texts)-empty_deseg}/{len(deseg_texts)} OK | {time.time()-t1:.1f}s")

    # Gemini entailment hypotheses
    t2 = time.time()
    ent_items = [{"question": r["questions"], "answer": r["answer_text"]}
                 for _, r in df.iterrows()]
    ent_hyps = batch_generate(ent_items, HYPOTHESIS_PROMPT,
                               output_dir / f"gemini_ent_cache_{split}.json")
    logger.info(f"[{split}] Entailment hypotheses done | {time.time()-t2:.1f}s")

    # Heuristic A+C contradiction (verb negation + entity swap from premise)
    t3 = time.time()
    heuristic_rng = random.Random(random_state)
    heuristic_counts: dict = {}
    con_pairs: list[tuple[int, str, str]] = []
    for i, row in df.iterrows():
        hyp = ent_hyps[i]
        if not hyp:
            continue
        modified, heuristic = apply_heuristics(
            hyp, rng=heuristic_rng, premise=row["sentence_deseg"],
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified:
            heuristic_counts[heuristic] = heuristic_counts.get(heuristic, 0) + 1
            con_pairs.append((i, modified, heuristic))
    logger.info(f"[{split}] {len(con_pairs):,}/{len(df):,} rows have a heuristic contradiction | {time.time()-t3:.1f}s")

    # Embeddings + neutral
    t4 = time.time()
    premise_texts = df["sentence_deseg"].tolist()
    emb = encode(premise_texts, cache_path=output_dir / f"embeddings_{split}.npy")
    sent_ids = df["sent_id"].tolist()
    neutral_pairs = select_neutral_pairs(emb, sent_ids, texts=premise_texts,
                                          random_state=random_state)
    logger.info(f"[{split}] Embeddings + neutral done | {time.time()-t4:.1f}s")

    out_rows = []
    rejected: dict[str, int] = {}

    for i, row in df.iterrows():
        hyp = ent_hyps[i]
        if not hyp:
            continue
        prem = row["sentence_deseg"]
        if not length_ok(prem, hyp):
            rejected["entailment"] = rejected.get("entailment", 0) + 1
            continue
        base = {
            "premise":        prem,
            "hypothesis":     hyp,
            "predicate":      row["predicate"],
            "predicate_type": row["type"],
            "dataset":        "qasem",
            "split":          split,
        }
        out_rows.append({**base, "id": f"qasem_{split}_{i:06d}_E", "label": "entailment"})

    for src_i, con_hyp, heuristic in con_pairs:
        row = df.iloc[src_i]
        prem = row["sentence_deseg"]
        if not length_ok(prem, con_hyp):
            rejected["contradiction"] = rejected.get("contradiction", 0) + 1
            continue
        out_rows.append({
            "id":                      f"qasem_{split}_{src_i:06d}_C",
            "premise":                  prem,
            "hypothesis":               con_hyp,
            "label":                    "contradiction",
            "contradiction_heuristic":  heuristic,
            "predicate":                row["predicate"],
            "predicate_type":           row["type"],
            "dataset":                  "qasem",
            "split":                    split,
        })

    for i, j in neutral_pairs:
        hyp = ent_hyps[j]
        if not hyp:
            continue
        row = df.iloc[i]
        prem = row["sentence_deseg"]
        if not length_ok(prem, hyp):
            rejected["neutral"] = rejected.get("neutral", 0) + 1
            continue
        out_rows.append({
            "id":             f"qasem_{split}_{i:06d}_N",
            "premise":         prem,
            "hypothesis":      hyp,
            "label":           "neutral",
            "predicate":       row["predicate"],
            "predicate_type":  row["type"],
            "dataset":         "qasem",
            "split":           split,
        })
    log_length_rejections(logger, rejected, split)

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "qasem")
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | total: {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="QASem Hebrew → NLI JSONL")
    parser.add_argument("--data_dir",      required=True, help="Path to QASem_Hebrew/")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--log_dir",       default="logs")
    parser.add_argument("--train_sample",  type=int, default=10000)
    parser.add_argument("--random_state",  type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: random N rows per split (0=disabled)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_dir), "06_qasem_convert")
    log_gpu_info(logger)
    logger.info(f"QASem conversion started | train_sample={args.train_sample}"
                + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    total = 0
    for split, fname in SPLIT_FILES.items():
        df = pd.read_csv(data_dir / fname)
        if split == "train" and args.train_sample and len(df) > args.train_sample:
            df = df.sample(n=args.train_sample, random_state=args.random_state).reset_index(drop=True)
            logger.info(f"[train] Subsampled to {len(df):,} rows")
        total += convert_split(split, df, output_dir, args.random_state, smoke=args.smoke)
    logger.info(f"QASem done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
