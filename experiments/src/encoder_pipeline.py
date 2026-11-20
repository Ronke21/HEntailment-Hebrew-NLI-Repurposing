#!/usr/bin/env python3
"""
encoder_pipeline.py — Encoder fine-tuning (3-class sequence classification) on a
repurposed NLI dataset, then evaluation on all 8 test sets.

Mirrors finetune_heb_nli/scripts/finetune_NLI_encoders.py but is fully
self-contained: it reads the unified premise/hypothesis/label JSONL directly
instead of going through the mrl_eval pipeline.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import sys

import numpy as np
import torch
import torch.distributed as tdist
import transformers
import ckpt_utils
from torch.utils.data import Dataset

import config
import data_utils
import metrics_utils

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = WORLD_SIZE > 1


class NLIPairDataset(Dataset):
    """Tokenised (premise, hypothesis) pairs for sequence classification."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int,
                 zero_token_type_ids: bool = False):
        self._data = []
        for ex in examples:
            enc = tokenizer(
                ex["premise"],
                ex["hypothesis"],
                max_length=max_length,
                truncation=True,
            )
            item = {k: enc[k] for k in enc.keys()}
            if zero_token_type_ids and "token_type_ids" in item:
                item["token_type_ids"] = [0] * len(item["token_type_ids"])
            item["labels"] = config.LABEL2ID[ex["label"]]
            self._data.append(item)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    y_true = [config.ID2LABEL[int(l)] for l in labels]
    y_pred = [config.ID2LABEL[int(p)] for p in preds]
    return {
        "macro_f1": metrics_utils.macro_f1(y_true, y_pred, config.LABELS_EN),
        "accuracy": metrics_utils.accuracy(y_true, y_pred),
    }


class LoggingCallback(transformers.TrainerCallback):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step=%d | %s", state.global_step, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("EVAL step=%d | %s", state.global_step, metrics)


def _repair_nonpersistent_rope_buffers(model, log: logging.Logger) -> int:
    """Recompute NeoBERT's RoPE tables after from_pretrained(). Returns #fixed.

    Why: without it dicta-il/neodictabert collapses to NaN (loss=0.0,
    grad_norm=nan from step one, every prediction a single class).

    modeling_neobert.py registers its rotary tables as NON-PERSISTENT buffers
    computed in __init__:

        cos, sin = precompute_freqs(hidden_size // num_attention_heads, max_length)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    persistent=False deliberately keeps them out of the checkpoint, because
    they are derivable constants. But transformers 5.x's fast-init path
    materialises modules without running that __init__ computation, and then
    only fills tensors that appear in the state_dict -- which these, by
    construction, never do. The tables are therefore left as raw
    uninitialised memory (observed leading values 1.4e-39 / 5.05e-42 --
    denormal garbage; freqs_cos[0] should be all 1.0), and RoPE poisons every
    layer from the first forward pass.

    This also explains why it looked host- and input-dependent and cost so
    long to find: uninitialised memory differs per machine and per run, so
    the same model NaN'd on one host and not another, and 9 of 16 grid units
    survived by luck. Do NOT gate this repair on "are the buffers NaN?" --
    a run was observed where they contained no NaN yet were still garbage and
    the output went NaN anyway. Always recompute.

    Written to be a no-op for every other architecture (nothing else in this
    roster exposes freqs_cos/freqs_sin), so it is safe on the shared path.
    """
    fixed = 0
    for name, mod in model.named_modules():
        if not (hasattr(mod, "freqs_cos") and hasattr(mod, "freqs_sin")):
            continue
        precompute = getattr(sys.modules.get(type(mod).__module__, None),
                             "precompute_freqs", None)
        if precompute is None:
            log.warning("RoPE repair: %s has freqs buffers but no precompute_freqs "
                        "in its module -- left as-is", name or "<root>")
            continue
        cfg = model.config
        dim = cfg.hidden_size // cfg.num_attention_heads
        cos, sin = precompute(dim, cfg.max_length)
        with torch.no_grad():
            mod.freqs_cos.copy_(cos.to(mod.freqs_cos.device, mod.freqs_cos.dtype))
            mod.freqs_sin.copy_(sin.to(mod.freqs_sin.device, mod.freqs_sin.dtype))
        fixed += 1
        log.info("RoPE repair: recomputed non-persistent freqs_cos/freqs_sin on %s "
                 "(dim=%d, max_length=%d)", name or "<root>", dim, cfg.max_length)
    return fixed


def run_encoder_experiment(
    model_name: str,
    train_spec: str,
    log: logging.Logger,
    seed: int = config.DEFAULT_SEED,
    max_per_dataset: int | None = None,
    keep_checkpoint: bool = False,
) -> dict:
    """Fine-tune *model_name* on *train_spec*, evaluate on all 8 test sets.

    Returns the metrics dict; also writes metrics.json + per-testset
    prediction files under results/<model>/train_<spec>/.
    """
    hp    = config.ENCODER_CLF
    flags = config.MODEL_FLAGS.get(model_name, {})
    out_run  = config.run_dir(model_name, train_spec, seed)
    out_ckpt = config.ckpt_dir(model_name, train_spec, seed)
    pred_dir = out_run / "predictions"
    for d in (out_run, out_ckpt, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    train_data, dev_data = data_utils.load_train_dev(train_spec, seed, max_per_dataset)
    test_sets = data_utils.load_all_test_sets()
    log.info("train_spec=%s  train=%d  dev=%d  test sets: %s",
             train_spec, len(train_data), len(dev_data),
             {k: len(v) for k, v in test_sets.items()})

    # ── tokenizer / datasets ─────────────────────────────────────────────────
    trust_remote_code = flags.get("trust_remote_code", False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=trust_remote_code
    )
    zero_tti = flags.get("zero_token_type_ids", False)

    # clamp max_length to what the model's position embeddings actually allow
    # (XLM-R/AlephBERT: 512; NeoDictaBERT: 4096) — long premises just truncate more
    model_cfg = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    mpe = getattr(model_cfg, "max_position_embeddings", None)
    max_length = hp["max_length"]
    if mpe:
        # RoBERTa-family reserves 2 positions (pad offset), so subtract 2 to be safe
        max_length = min(max_length, mpe - 2)
    log.info("Tokenising (max_length=%d [config %d, model positions %s], "
             "zero_token_type_ids=%s)", max_length, hp["max_length"], mpe, zero_tti)
    train_ds = NLIPairDataset(train_data, tokenizer, max_length, zero_tti)
    dev_ds   = NLIPairDataset(dev_data,   tokenizer, max_length, zero_tti)

    # ── model ────────────────────────────────────────────────────────────────
    model_kwargs: dict = dict(
        num_labels=len(config.LABELS_EN),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True,
        trust_remote_code=trust_remote_code,
    )
    if flags.get("attn_implementation"):
        model_kwargs["attn_implementation"] = flags["attn_implementation"]
    dm_override = os.environ.get("DEVICE_MAP_OVERRIDE") or flags.get("device_map")
    if IS_DISTRIBUTED:
        if dm_override:
            log.warning("Distributed launch (WORLD_SIZE=%d) — ignoring "
                        "DEVICE_MAP_OVERRIDE/device_map flag (%s); Trainer places "
                        "the model on this rank's own GPU for DDP instead",
                        WORLD_SIZE, dm_override)
    elif dm_override:
        model_kwargs["device_map"] = dm_override

    _init_fn = getattr(config, "INIT_FROM", None)
    _init_path = _init_fn(model_name, train_spec, seed) if callable(_init_fn) else None
    _load_src = str(_init_path) if _init_path else model_name
    if _init_path:
        import pathlib as _pl
        if not any((_pl.Path(_init_path) / f).exists()
                   for f in ("model.safetensors", "pytorch_model.bin")):
            raise FileNotFoundError(f"INIT_FROM checkpoint missing: {_init_path}")
        log.info("INIT_FROM: loading fine-tuned weights from %s", _init_path)
    log.info("Loading model %s (num_labels=3)", _load_src)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        _load_src, **model_kwargs
    )
    _repair_nonpersistent_rope_buffers(model, log)
    log.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    force_fp32 = os.environ.get("FORCE_FP32") == "1"
    use_bf16 = (not force_fp32) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = (not force_fp32) and torch.cuda.is_available() and not use_bf16

    train_bs = max(1, hp["batch_size"] // 2) if force_fp32 else hp["batch_size"]
    eval_bs = max(1, hp["eval_batch_size"] // 4) if force_fp32 else hp["eval_batch_size"]
    use_grad_ckpt = force_fp32 and getattr(model, "supports_gradient_checkpointing", False)

    # ── training setup ───────────────────────────────────────────────────────
    max_steps = config.compute_max_steps(
        len(train_data), train_bs, 1,
        hp["max_epochs"], hp["max_steps_cap"], hp["min_steps"])
    eval_steps = config.compute_eval_steps(
        len(train_data), train_bs, 1, config.eval_per_epoch_for(train_spec, hp),
        min_eval_steps=hp.get("min_eval_steps", 50))
    log.info("max_steps=%d  eval_steps=%d  train_bs=%d  eval_bs=%d",
             max_steps, eval_steps, train_bs, eval_bs)

    training_args = transformers.TrainingArguments(
        output_dir                 = str(out_ckpt),
        max_steps                  = max_steps,
        per_device_train_batch_size= train_bs,
        per_device_eval_batch_size = eval_bs,
        gradient_checkpointing     = use_grad_ckpt,
        learning_rate              = hp["lr"],
        warmup_ratio               = hp["warmup_ratio"],
        weight_decay               = hp["weight_decay"],
        eval_strategy              = "steps",
        eval_steps                 = eval_steps,
        save_strategy              = "steps",
        save_steps                 = eval_steps,
        save_total_limit           = hp["save_total_limit"],
        load_best_model_at_end     = True,
        metric_for_best_model      = "macro_f1",
        greater_is_better          = True,
        logging_steps              = 50,
        report_to                  = [],
        seed                       = seed,
        data_seed                  = seed,
        bf16                       = use_bf16,
        fp16                       = use_fp16,
        auto_find_batch_size       = True,
    )

    trainer = transformers.Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = dev_ds,
        data_collator   = transformers.DataCollatorWithPadding(tokenizer),
        compute_metrics = _compute_metrics,
        callbacks       = [
            LoggingCallback(log),
            transformers.EarlyStoppingCallback(
                early_stopping_patience=hp["early_stop_patience"]),
        ],
    )

    # resume mid-training if a previous run of this exact (model, spec, seed)
    # crashed — HF restores optimizer/scheduler/step from the checkpoint
    last_ckpt = ckpt_utils.last_complete_checkpoint(out_ckpt, log)
    if last_ckpt:
        log.info("Resuming training from checkpoint: %s", last_ckpt)
    log.info("Starting training …")
    trainer.train(resume_from_checkpoint=last_ckpt)
    log.info("Training done. best_checkpoint=%s dev_macro_f1=%.4f",
             trainer.state.best_model_checkpoint, trainer.state.best_metric or 0.0)

    _evals = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    _grads = [h["grad_norm"] for h in trainer.state.log_history if "grad_norm" in h]
    _nan = lambda xs: bool(xs) and all(v != v for v in xs)   # v != v  <=> isnan(v)
    if _nan(_evals) or _nan(_grads):
        raise RuntimeError(
            f"training never converged: eval_loss NaN on all {len(_evals)} evals, "
            f"grad_norm NaN on all {len(_grads)} logged steps -- model did not train. "
            f"Refusing to write metrics.json for {model_name}/{train_spec}/seed {seed}.")

    if train_spec in getattr(config, "INIT_SOURCE_SPECS", ()):
        _f1s = [h["eval_macro_f1"] for h in trainer.state.log_history
                if "eval_macro_f1" in h]
        _best = max(_f1s) if _f1s else None
        if _best is not None and _best < 0.40:      # 3-class chance = 0.333
            raise RuntimeError(
                f"degenerate training: best eval_macro_f1 {_best:.4f} over "
                f"{len(_f1s)} evals is at chance level -- the model converged to "
                f"a constant prediction. This spec is an INIT_FROM source "
                f"refusing to keep it for {model_name}/{train_spec}/seed {seed}.")

    # full loss/metric curve for later analysis (rank-0 only — every rank has
    # an identical trainer.state after DDP training, so no data is lost)
    if trainer.is_world_process_zero():
        with open(out_run / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    dev_result = trainer.evaluate(eval_dataset=dev_ds)
    log.info("Dev metrics: %s", dev_result)

    # ── evaluate best model on all 8 test sets ───────────────────────────────
    test_metrics: dict[str, dict] = {}
    y_true_all: list = []
    y_pred_all: list = []
    for ds_name, examples in test_sets.items():
        pred_file = pred_dir / f"{ds_name}.jsonl"
        y_true = [ex["label"] for ex in examples]

        cached = data_utils.load_cached_predictions(pred_file, examples)
        if cached is not None:
            log.info("Test set %s: reusing cached predictions (%d) — eval resume",
                     ds_name, len(cached))
            y_pred = cached
        else:
            log.info("Evaluating on test set: %s (n=%d)", ds_name, len(examples))
            test_ds = NLIPairDataset(examples, tokenizer, max_length, zero_tti)
            # trainer.predict() is itself a collective, DDP-parallelised op —
            # every rank calls it and receives the same full gathered
            # predictions back, so only the FILE WRITE below needs gating
            pred_out = trainer.predict(test_ds)
            preds_int = np.argmax(pred_out.predictions, axis=-1)
            y_pred = [config.ID2LABEL[int(p)] for p in preds_int]
            if trainer.is_world_process_zero():
                with open(pred_file, "w", encoding="utf-8") as f:
                    for ex, pred in zip(examples, y_pred):
                        f.write(json.dumps({"id": ex["id"], "gold": ex["label"],
                                            "prediction": pred}, ensure_ascii=False) + "\n")

        test_metrics[ds_name] = metrics_utils.full_report(y_true, y_pred, config.LABELS_EN)
        log.info("  %s → %s", ds_name, test_metrics[ds_name])
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    # ── save final metrics + best model ──────────────────────────────────────
    results = {
        "model"        : model_name,
        "architecture" : "encoder",
        "approach"     : "sequence_classification",
        "train_dataset": train_spec,
        "seed"         : seed,
        "n_train"      : len(train_data),
        "n_dev"        : len(dev_data),
        "max_steps"    : max_steps,
        "max_length"   : max_length,
        "hyperparams"  : dict(hp),
        "model_flags"  : dict(flags),
        "resumed_from" : last_ckpt,
        "finished_at"  : datetime.datetime.now().isoformat(timespec="seconds"),
        "versions"     : {"transformers": transformers.__version__,
                          "torch": torch.__version__},
        "dev"          : {
            "macro_f1": round(dev_result.get("eval_macro_f1", 0.0), 4),
            "accuracy": round(dev_result.get("eval_accuracy", 0.0), 4),
        },
        "test"         : test_metrics,
        # "together" scores across all evaluated test sets — pooled: one
        # score over the concatenated union; averaged: mean of the per-set
        # scores above. Different statistics, both kept (see metrics_utils.
        # average_reports's docstring for why).
        "test_pooled"  : metrics_utils.full_report(y_true_all, y_pred_all, config.LABELS_EN),
        "test_averaged": metrics_utils.average_reports(test_metrics),
    }
    # rank-0 only: metrics.json/checkpoint writes below would otherwise race
    # across ranks (concurrent writers to the same path, or one rank's
    # rmtree racing another's)
    if trainer.is_world_process_zero():
        with open(out_run / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        log.info("Metrics written → %s", out_run / "metrics.json")

        if keep_checkpoint:
            best_dir = out_ckpt / "best"
            trainer.save_model(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            log.info("Best model saved → %s", best_dir)
        else:
            # free disk: intermediate checkpoints are not needed after eval
            import shutil
            for sub in out_ckpt.glob("checkpoint-*"):
                shutil.rmtree(sub, ignore_errors=True)
            log.info("Intermediate checkpoints removed (pass --keep-checkpoints to keep)")

    # other ranks wait here so a later unit's collective trainer.train() (if
    # this process's queue has more than one) can't start out of step
    if IS_DISTRIBUTED and tdist.is_initialized():
        tdist.barrier()

    return results
