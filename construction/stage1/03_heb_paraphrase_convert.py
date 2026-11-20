"""
Task 5 — Hebrew Paraphrase Dataset Conversion
V2: (original, paraphrase) E+C+N | V3: (paraphrase, original) E+C+N (bidirectional).
V1 (Gemini title as premise) removed — Gemma eval confirmed C-F1=0.000, no discriminative signal.
Contradiction: 21 heuristics (all-fire-then-choose). NeoDictaBERT neutral.
80/10/10 split stratified by type. Skips splits whose output JSONL already exists.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils.embed_utils import encode, select_neutral_pairs
from utils.contradiction_utils import apply_heuristics
from utils.logging_utils import (setup_logging, is_split_done, log_gpu_info,
                                  log_label_stats, log_examples, log_heuristic_breakdown,
                                  length_ok, log_length_rejections)

import logging
logger = logging.getLogger(__name__)

# Paraphrase pairs are semantically near-identical; only high-salience heuristics
# produce a change visible enough for the model to detect as contradiction.
_PARA_ALLOWED = {
    "verb_negation", "quantifier_flip",
    "number_substitution", "number_substitution_heb",
    "superlative_flip", "existential_negation",
}


def convert_split(split: str, df: pd.DataFrame, output_dir: Path,
                  all_embeddings: np.ndarray,
                  global_indices: list[int], random_state: int) -> int:
    if is_split_done(output_dir, split):
        return 0

    t0 = time.time()
    logger.info(f"[{split}] {len(df)} pairs")
    heuristic_rng = random.Random(random_state)
    heuristic_counts: dict = {}

    originals   = df["original_text"].tolist()
    paraphrases = df["paraphrase"].tolist()
    types       = df["type"].tolist()
    golds       = df["gold"].tolist()

    sub_emb = all_embeddings[global_indices]
    neutral_pairs = select_neutral_pairs(sub_emb, list(range(len(df))), random_state=random_state)

    out_rows = []
    rejected: dict[str, int] = {}

    for i in range(len(df)):
        orig   = originals[i]
        para   = paraphrases[i]
        base   = {"dataset": "heb_paraphrase", "split": split,
                  "type": types[i], "gold": bool(golds[i])}

        # V2: original as premise, paraphrase as hypothesis
        if length_ok(orig, para):
            out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_E_v2",
                             "premise": orig, "hypothesis": para,
                             "label": "entailment", "version": "v2"})
        else:
            rejected["entailment_v2"] = rejected.get("entailment_v2", 0) + 1

        modified_v2, heuristic_v2 = apply_heuristics(
            para, rng=heuristic_rng, premise=orig, allowed=_PARA_ALLOWED,
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified_v2:
            if length_ok(orig, modified_v2):
                heuristic_counts[heuristic_v2] = heuristic_counts.get(heuristic_v2, 0) + 1
                out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_C_v2",
                                 "premise": orig, "hypothesis": modified_v2,
                                 "label": "contradiction", "version": "v2",
                                 "contradiction_heuristic": heuristic_v2})
            else:
                rejected["contradiction_v2"] = rejected.get("contradiction_v2", 0) + 1

        # V3: paraphrase as premise, original as hypothesis (reverse direction)
        if length_ok(para, orig):
            out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_E_v3",
                             "premise": para, "hypothesis": orig,
                             "label": "entailment", "version": "v3"})
        else:
            rejected["entailment_v3"] = rejected.get("entailment_v3", 0) + 1

        modified_v3, heuristic_v3 = apply_heuristics(
            orig, rng=heuristic_rng, premise=para, allowed=_PARA_ALLOWED,
            heuristic_counts=heuristic_counts, max_fraction=0.25)
        if modified_v3:
            if length_ok(para, modified_v3):
                heuristic_counts[heuristic_v3] = heuristic_counts.get(heuristic_v3, 0) + 1
                out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_C_v3",
                                 "premise": para, "hypothesis": modified_v3,
                                 "label": "contradiction", "version": "v3",
                                 "contradiction_heuristic": heuristic_v3})
            else:
                rejected["contradiction_v3"] = rejected.get("contradiction_v3", 0) + 1

    for i, j in neutral_pairs:
        base = {"dataset": "heb_paraphrase", "split": split,
                "type": types[i], "gold": bool(golds[i])}
        if length_ok(originals[i], paraphrases[j]):
            out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_N_v2",
                             "premise": originals[i], "hypothesis": paraphrases[j],
                             "label": "neutral", "version": "v2"})
        else:
            rejected["neutral_v2"] = rejected.get("neutral_v2", 0) + 1
        if length_ok(paraphrases[i], originals[j]):
            out_rows.append({**base, "id": f"heb_para_{split}_{i:05d}_N_v3",
                             "premise": paraphrases[i], "hypothesis": originals[j],
                             "label": "neutral", "version": "v3"})
        else:
            rejected["neutral_v3"] = rejected.get("neutral_v3", 0) + 1
    log_length_rejections(logger, rejected, split)

    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log_label_stats(logger, out_rows, split, "heb_paraphrase")
    log_heuristic_breakdown(logger, out_rows, split)
    log_examples(logger, out_rows, split)
    logger.info(f"[{split}] Wrote {len(out_rows):,} rows → {out_path} | {time.time()-t0:.1f}s")
    return len(out_rows)


def main():
    parser = argparse.ArgumentParser(description="Heb-Paraphrase → NLI JSONL")
    parser.add_argument("--data_file",    required=True, help="Path to Dataset.json")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--log_dir",      default="logs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke-test mode: limit to N pairs total (0=disabled)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.log_dir), "03_heb_paraphrase_convert")
    log_gpu_info(logger)
    logger.info("Heb-Paraphrase conversion started" + (f" | SMOKE={args.smoke}" if args.smoke else ""))

    t_start = time.time()
    data = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    df   = pd.DataFrame(data)
    if "original_text" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "original_text"})
    if args.smoke > 0:
        df = df.sample(min(args.smoke, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"SMOKE MODE: random sample of {len(df)} pairs")
    logger.info(f"Loaded {len(df):,} pairs | columns: {df.columns.tolist()}")
    logger.info(f"Type breakdown: {df['type'].value_counts().to_dict()}")
    logger.info(f"Gold: {df['gold'].sum()} | Non-gold: {(~df['gold']).sum()}")

    # Embeddings (one-time, over all originals)
    t2 = time.time()
    all_embeddings = encode(df["original_text"].tolist(), cache_path=output_dir / "embeddings.npy")
    logger.info(f"Embeddings done: {all_embeddings.shape} | {time.time()-t2:.1f}s")

    # Split (track global indices)
    df["_global_idx"] = range(len(df))
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=args.random_state,
                                          stratify=df["type"])
    dev_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=args.random_state,
                                          stratify=temp_df["type"])
    splits = {
        "train": train_df.reset_index(drop=True),
        "dev":   dev_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
    }
    logger.info(f"Split sizes: train={len(train_df)} dev={len(dev_df)} test={len(test_df)}")

    total = 0
    for split, split_df in splits.items():
        global_indices = split_df["_global_idx"].tolist()
        clean_df       = split_df.drop(columns=["_global_idx"]).reset_index(drop=True)
        total += convert_split(split, clean_df, output_dir,
                               all_embeddings, global_indices, args.random_state)

    logger.info(f"Heb-Paraphrase done. Total rows: {total:,} | wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
