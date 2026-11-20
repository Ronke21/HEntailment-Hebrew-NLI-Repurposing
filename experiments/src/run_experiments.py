#!/usr/bin/env python3
"""
run_experiments.py — Orchestrator for the repurposed-data transfer benchmark.

For each (model, train_dataset, seed) triple in the grid:
    1. fine-tune the model on that dataset's train split (dev for selection)
       — encoders  → 3-class sequence classification head
       — decoders  → LoRA SFT with Hebrew label targets
    2. evaluate the best checkpoint on ALL 8 test sets:
       winograd, heq, qasem, hesum, hebsummaries, heb_paraphrase, hebnli, lchaim

train_dataset may also be "all" (concatenation of the 6 repurposed datasets)
or "all_balanced" (same, capped at 10k per dataset so heq/qasem don't dominate).
Default seeds: 42,43,44,45,46 — build_summary.py reports mean ± std.

Resume-safe at three levels:
  * completed runs   — skipped if seed_<N>/metrics.json exists
  * crashed training — resumes from the latest checkpoint in checkpoints/
  * crashed eval     — per-testset prediction files are reused, only the
                       missing test sets are re-scored
Use --rerun to wipe a run's artifacts and start it from scratch.

Conda environment: heb_nli_mrl_eval  (or heb_nli_b200 on B200 nodes)

Usage (from Hebrew_NLI/ — EXPERIMENT_ROOT required, see below):
    conda activate heb_nli_mrl_eval
    cd .
    export EXPERIMENT_ROOT=finetune_repurposed_data

    # Full grid — default models × (6 datasets + "all") × 5 seeds:
    CUDA_VISIBLE_DEVICES=0 python finetune_repurposed_data/src/run_experiments.py

    # One model on one dataset, single seed (quick exploration):
    python finetune_repurposed_data/src/run_experiments.py \\
        --models "dicta-il/neodictabert" --train-datasets winograd --seeds 42

    # One model trained on all tasks together (with per-dataset cap):
    python finetune_repurposed_data/src/run_experiments.py \\
        --models "google/gemma-2-9b-it" --train-datasets all --max-per-dataset 10000

Aliases accepted for dataset names: samqa→qasem, hepar→heb_paraphrase,
hesummaries→hebsummaries.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import sys
import traceback

def _early_experiment_root() -> str:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--experiment-root" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--experiment-root="):
            return a.split("=", 1)[1]
    return os.environ.get("EXPERIMENT_ROOT", "")

EXPERIMENT_ROOT = _early_experiment_root()
if not EXPERIMENT_ROOT:
    sys.exit("ERROR: --experiment-root (or EXPERIMENT_ROOT env var) must be set, e.g.\n"
             "  python finetune_repurposed_data/src/run_experiments.py "
             "--experiment-root finetune_repurposed_data/experiments/subset3k ...")
sys.path.insert(0, str(pathlib.Path(EXPERIMENT_ROOT) / "src"))

import config  # noqa: E402

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
IS_DISTRIBUTED = WORLD_SIZE > 1
IS_MAIN = RANK == 0


def make_logger(model_name: str, train_spec: str,
                seed: int) -> tuple[logging.Logger, pathlib.Path]:
    """Per-run logger writing to logs/<model>__train_<spec>__seed_<N>.log and stdout.

    Under a distributed (torchrun) launch, only rank 0 gets real handlers —
    every rank would otherwise open the same log_path in mode="w" and
    truncate/interleave each other's writes."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (config.LOGS_DIR /
                f"{config.safe_name(model_name)}__train_{train_spec}__seed_{seed}.log")
    log = logging.getLogger(f"{model_name}::{train_spec}::{seed}")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()
    if not IS_MAIN:
        log.addHandler(logging.NullHandler())
        return log, log_path
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log, log_path


def detect_architecture(model_name: str) -> str:
    """Return 'encoder' or 'decoder' — from the known model lists first, then
    the HF config. NB: a bare 'LMHead' in architectures is NOT decoder evidence
    (BERT-style masked-LM heads are also called *LMHeadModel, e.g. NeoBERT)."""
    if model_name in config.ENCODER_MODELS:
        return "encoder"
    if model_name in config.DECODER_MODELS:
        return "decoder"

    import transformers
    flags = config.MODEL_FLAGS.get(model_name, {})
    cfg = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=flags.get("trust_remote_code", False))
    if getattr(cfg, "is_encoder_decoder", False):
        raise ValueError(
            f"{model_name} is an encoder-decoder model — not supported by this "
            "pipeline (use the mrl_eval pipeline in finetune_heb_nli for mT5).")
    archs = getattr(cfg, "architectures", None) or []
    if any("ForCausalLM" in a or "ForConditionalGeneration" in a for a in archs):
        return "decoder"
    if any("MaskedLM" in a or "ForSequenceClassification" in a
           or "ForPreTraining" in a for a in archs):
        return "encoder"
    # GPT-style decoders whose only head class is *LMHeadModel (e.g. GPT2LMHeadModel)
    if any(a.endswith("LMHeadModel") for a in archs) and getattr(cfg, "is_decoder", False):
        return "decoder"
    return "encoder"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-root", default=EXPERIMENT_ROOT, dest="experiment_root",
                        help="Absolute path to the experiment directory (config/data/"
                             "results live under here). Already consumed by "
                             "_early_experiment_root() above before this parser even "
                             "runs — registered here only so argparse doesn't reject "
                             "it as unrecognized.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated HF model IDs (default: config.DEFAULT_MODELS)")
    parser.add_argument("--train-datasets", default=None, dest="train_datasets",
                        help="Comma-separated train specs out of "
                             f"{config.TRAIN_SPECS_DEFAULT} "
                             "(default: all of them). Aliases like samqa/hepar accepted.")
    parser.add_argument("--arch", choices=["auto", "encoder", "decoder"], default="auto",
                        help="Force architecture routing (default: auto-detect from config)")
    parser.add_argument("--seeds", default=None,
                        help="Comma-separated seeds (default: "
                             f"{','.join(map(str, config.DEFAULT_SEEDS))}). "
                             "Use a single seed for quick exploration.")
    parser.add_argument("--max-per-dataset", type=int, default=None, dest="max_per_dataset",
                        help="Cap each dataset's train contribution in the 'all' mixture")
    parser.add_argument("--keep-checkpoints", action="store_true", dest="keep_checkpoints",
                        help="Save the best model/adapter after eval (default: delete "
                             "checkpoints once metrics are written)")
    parser.add_argument("--rerun", action="store_true",
                        help="Wipe each selected run's artifacts (results + "
                             "checkpoints) and run it again from scratch")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: tiny data slices (64/32/16 per split), "
                             "~8 training steps, results under results_smoke/ — "
                             "verifies the full pipeline in minutes, numbers are junk")
    parser.add_argument("--zero-shot", action="store_true", dest="zero_shot",
                        help="Instead of fine-tuning, evaluate each DECODER model "
                             "zero-shot on all 8 test sets (no-training baseline; "
                             "runs once, ignores --train-datasets/--seeds; encoders "
                             "are skipped — an untrained classification head is random)")
    args = parser.parse_args()

    if args.smoke:
        config.enable_smoke_mode()
        print("[SMOKE] smoke-test mode: data caps "
              f"{config.SMOKE_DATA_CAPS}, ~8 steps, results → {config.RESULTS_DIR}")

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else config.DEFAULT_MODELS)
    if args.train_datasets and not args.zero_shot:
        train_specs = [config.resolve_dataset(s)
                       for s in args.train_datasets.split(",") if s.strip()]
    else:
        train_specs = config.TRAIN_SPECS_DEFAULT
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else config.DEFAULT_SEEDS)

    t0 = datetime.datetime.now()
    print(f"\n{'#'*76}\n"
          f"# Repurposed-data transfer benchmark\n"
          f"# Started        : {t0.isoformat()}\n"
          f"# Models         : {models}\n"
          f"# Train datasets : {train_specs}\n"
          f"# Seeds          : {seeds}\n"
          f"# Eval test sets : {config.EVAL_DATASETS}\n"
          f"# Data root      : {config.DATA_ROOT}\n"
          f"{'#'*76}\n")

    # ── data-presence check up front ─────────────────────────────────────────
    missing = []
    for ds in set(config.EVAL_DATASETS):
        for split in ("train", "dev", "test"):
            p = config.DATA_ROOT / ds / f"{split}.jsonl"
            if not p.exists():
                missing.append(str(p))
    if missing:
        print("[FATAL] Missing data files:\n  " + "\n  ".join(missing))
        sys.exit(1)

    all_results: dict = {}

    # ── zero-shot baseline mode: no training, one deterministic pass ─────────
    if args.zero_shot:
        for model_name in models:
            key = f"{model_name} | zero-shot"
            print(f"\n{'='*76}\n  {key}\n{'='*76}")
            zs_file = (config.RESULTS_DIR / config.safe_name(model_name)
                       / "zero_shot" / "metrics.json")
            if not args.rerun and zs_file.exists():
                print(f"  [SKIP] metrics.json already exists: {zs_file}")
                all_results[key] = {"status": "success",
                                    "results": json.load(open(zs_file))}
                continue
            log, log_path = make_logger(model_name, "zeroshot", 0)
            try:
                if detect_architecture(model_name) != "decoder":
                    print(f"  [SKIP] {model_name} is an encoder — zero-shot with an "
                          "untrained classification head is meaningless")
                    continue
                import decoder_pipeline
                results = decoder_pipeline.run_decoder_zero_shot(model_name, log)
                all_results[key] = {"status": "success", "results": results}
            except Exception as exc:
                tb = traceback.format_exc()
                log.error("FAILED: %s\n%s", exc, tb)
                print(f"[ERROR] {key} failed — see {log_path}: {exc}")
                all_results[key] = {"status": "failed", "error": str(exc),
                                    "traceback": tb}
        if IS_MAIN:
            _print_summary(all_results, t0)
        return

    for model_name in models:
        for train_spec in train_specs:
            for seed in seeds:
                key = f"{model_name} | train={train_spec} | seed={seed}"
                print(f"\n{'='*76}\n  {key}\n{'='*76}")

                if not args.rerun and config.is_run_complete(model_name, train_spec, seed):
                    mfile = config.run_dir(model_name, train_spec, seed) / "metrics.json"
                    print(f"  [SKIP] metrics.json already exists: {mfile}")
                    try:
                        all_results[key] = {"status": "success",
                                            "results": json.load(open(mfile))}
                    except Exception:
                        pass
                    continue

                if args.rerun:
                    # wipe old artifacts so checkpoint/eval resume caches
                    # can't leak into the fresh run
                    import shutil
                    for stale in (config.run_dir(model_name, train_spec, seed),
                                  config.ckpt_dir(model_name, train_spec, seed)):
                        if stale.exists():
                            shutil.rmtree(stale, ignore_errors=True)
                            print(f"  [RERUN] wiped {stale}")

                log, log_path = make_logger(model_name, train_spec, seed)
                log.info("MODEL=%s  TRAIN=%s  SEED=%d", model_name, train_spec, seed)

                try:
                    arch = args.arch
                    if arch == "auto":
                        arch = detect_architecture(model_name)
                    log.info("Architecture: %s", arch)

                    if arch == "encoder":
                        import encoder_pipeline
                        results = encoder_pipeline.run_encoder_experiment(
                            model_name, train_spec, log, seed=seed,
                            max_per_dataset=args.max_per_dataset,
                            keep_checkpoint=getattr(config, "KEEP_ENCODER_CHECKPOINT", False))
                    else:
                        import decoder_pipeline
                        results = decoder_pipeline.run_decoder_experiment(
                            model_name, train_spec, log, seed=seed,
                            max_per_dataset=args.max_per_dataset,
                            keep_checkpoint=True)

                    all_results[key] = {"status": "success", "results": results}
                    log.info("FINISHED SUCCESSFULLY")

                except Exception as exc:
                    tb = traceback.format_exc()
                    log.error("FAILED: %s", exc)
                    log.error("Traceback:\n%s", tb)
                    print(f"\n[ERROR] {key} failed — see {log_path}\n  Reason: {exc}\n"
                          "Continuing to next run …\n")
                    all_results[key] = {"status": "failed", "error": str(exc),
                                        "traceback": tb}
                finally:
                    # free GPU memory between runs
                    import gc
                    import torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    if IS_MAIN:
        _print_summary(all_results, t0)


def _print_summary(all_results: dict, t0: datetime.datetime) -> None:
    """Write grid_summary.json and print the per-run macro-F1 table."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.RESULTS_DIR / "grid_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    elapsed = datetime.datetime.now() - t0
    print(f"\n\n{'#'*76}")
    print("# SUMMARY — test macro-F1 per evaluation set (per run)")
    print(f"{'#'*76}")
    eval_cols = config.EVAL_DATASETS
    hdr = f"{'Run':<66} " + " ".join(f"{c[:9]:>9}" for c in eval_cols)
    print(hdr)
    print("-" * len(hdr))
    for key, entry in all_results.items():
        if entry.get("status") == "success":
            test = entry["results"].get("test", {})
            cells = " ".join(
                f"{test.get(c, {}).get('macro_f1', float('nan')):>9.4f}"
                if isinstance(test.get(c, {}).get("macro_f1"), (int, float)) else f"{'N/A':>9}"
                for c in eval_cols)
            print(f"{key:<66} {cells}")
        else:
            print(f"{key:<66} FAILED ← {entry.get('error', '?')[:60]}")

    print(f"\nTotal time  : {elapsed}")
    print(f"Grid summary: {summary_path}")
    print(f"Logs dir    : {config.LOGS_DIR}")
    print(f"Results dir : {config.RESULTS_DIR}")
    print("\nAggregate across seeds (mean ± std matrices) with:\n"
          f"  EXPERIMENT_ROOT={EXPERIMENT_ROOT} "
          "python finetune_repurposed_data/scripts/build_summary.py\n")


if __name__ == "__main__":
    main()
