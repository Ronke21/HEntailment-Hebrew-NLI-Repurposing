"""Shared logging setup and GPU/stats helpers."""

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: Path, script_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(script_name)
    logger.info(f"Log file: {log_file}")
    return logger


def log_gpu_info(logger: logging.Logger) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            logger.info(f"GPU: {torch.cuda.get_device_name(dev)} | "
                        f"Memory: {torch.cuda.get_device_properties(dev).total_memory // 1024**3} GB")
        else:
            logger.warning("CUDA not available — embedding step will be slow on CPU")
    except ImportError:
        logger.warning("torch not installed")


def is_split_done(output_dir: Path, split: str) -> bool:
    """Return True if the split JSONL exists and is non-empty."""
    p = output_dir / f"{split}.jsonl"
    if p.exists() and p.stat().st_size > 0:
        n = sum(1 for _ in p.open(encoding="utf-8"))
        logging.getLogger().info(f"[{split}] Already done ({n:,} rows) — skipping.")
        return True
    return False


def log_label_stats(logger: logging.Logger, rows: list[dict], split: str, dataset: str) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        lab = r.get("label", "unknown")
        counts[lab] = counts.get(lab, 0) + 1
    total = sum(counts.values())
    logger.info(
        f"[{split}] {dataset} label distribution ({total:,} rows): "
        + "  ".join(f"{k}:{v:,} ({100*v/total:.1f}%)" for k, v in sorted(counts.items()))
    )


def log_examples(logger: logging.Logger, rows: list[dict], split: str, n: int = 3) -> None:
    """Log n random examples per label."""
    by_label: dict[str, list] = {}
    for r in rows:
        lab = r.get("label", "unknown")
        by_label.setdefault(lab, []).append(r)
    rng = random.Random(0)
    for label, label_rows in sorted(by_label.items()):
        sample = rng.sample(label_rows, min(n, len(label_rows)))
        for ex in sample:
            premise = ex.get("premise", ex.get("premise_full", ""))[:100]
            hyp = ex.get("hypothesis", "")[:100]
            extra = ex.get("version", ex.get("contradiction_heuristic", ""))
            logger.info(f"  [{label}]{(' '+extra) if extra else ''} | P: {premise!r} | H: {hyp!r}")


def length_ok(premise: str, hypothesis: str) -> bool:
    """Return False only if hypothesis is more than 10 tokens longer than premise."""
    return len(hypothesis.split()) <= len(premise.split()) + 10


def log_length_rejections(logger: logging.Logger, rejected: dict, split: str) -> None:
    total = sum(rejected.values())
    if total:
        detail = "  ".join(f"{k}:{v}" for k, v in sorted(rejected.items()))
        logger.info(f"[{split}] Length-filter rejected {total} rows: {detail}")
    else:
        logger.info(f"[{split}] Length-filter: 0 rows rejected")


def log_heuristic_breakdown(logger: logging.Logger, rows: list[dict], split: str) -> None:
    c_rows = [r for r in rows if r.get("label") == "contradiction" and "contradiction_heuristic" in r]
    if not c_rows:
        return
    breakdown: dict[str, int] = {}
    for r in c_rows:
        h = r["contradiction_heuristic"]
        breakdown[h] = breakdown.get(h, 0) + 1
    logger.info(f"[{split}] Contradiction heuristic breakdown: " +
                "  ".join(f"{k}:{v}" for k, v in sorted(breakdown.items())))
