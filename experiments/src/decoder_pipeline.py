#!/usr/bin/env python3
"""
decoder_pipeline.py — Decoder LLM LoRA fine-tuning on a repurposed NLI dataset,
then generation-based evaluation on all 8 test sets.

Self-contained equivalent of the mrl_eval finetune→generate→evaluate pipeline
used in finetune_heb_nli/scripts/finetune_NLI_decoders.py:
  * prompt   : "משפט 1: {premise} משפט 2: {hypothesis}\nתשובה:"  (mrl_eval HebNLI format)
  * target   : Hebrew label word (היסק / סתירה / ניטרלי)
  * training : LoRA r=16 alpha=32 all-linear, bf16, loss on target tokens only
  * eval     : batched greedy generation + label normalisation (handles <think> blocks)
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.distributed as tdist
import transformers
import ckpt_utils
from torch.utils.data import Dataset

import config
import data_utils
import metrics_utils

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
IS_DISTRIBUTED = WORLD_SIZE > 1


def _autocast_dtype() -> torch.dtype:
    """bf16 where the GPU supports it (Ampere+: A100/B200), else fp16 —
    needed for V100 lanes (Volta has no native bf16 tensor cores)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _wrap_chat_template(prompt: str, tokenizer, template_kwargs: dict | None = None) -> str:
    """Wrap a plain-text instruction prompt in the tokenizer's own chat
    template. some models (google/gemma-4-31B-it) produce
    ~100% incoherent zero-shot output on the raw text this pipeline
    otherwise uses uniformly for every decoder — verified live that this
    wrapping alone fixes it, no fine-tuning needed. Gated per-model via
    MODEL_FLAGS["use_chat_template"] so already-working models' prompts
    (and any already-tokenized training caches) are untouched.

    *template_kwargs* (MODEL_FLAGS["chat_template_kwargs"]) is
    passed through to the Jinja render context, for reasoning models whose
    template opens a <think> block by default — with max_new_tokens=8 the
    whole generation is then the first few tokens of the reasoning trace and
    never reaches a label. Opt-in per model so the models already working
    under the plain template render byte-identically."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, **(template_kwargs or {}))


def build_prompt(ex: dict, tokenizer=None, max_prompt_tokens: int | None = None,
                 use_chat_template: bool = False, chat_template_kwargs: dict | None = None) -> str:
    """Render the prompt; if it exceeds *max_prompt_tokens*, truncate the
    PREMISE (keep its start) so the hypothesis and answer cue always survive —
    hesum/hebsummaries/lchaim premises are full articles."""
    prompt = config.DECODER_PROMPT_TEMPLATE.format(
        premise=ex["premise"], hypothesis=ex["hypothesis"])
    if tokenizer is None or max_prompt_tokens is None:
        return (_wrap_chat_template(prompt, tokenizer, chat_template_kwargs)
                if use_chat_template else prompt)
    if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) <= max_prompt_tokens:
        return (_wrap_chat_template(prompt, tokenizer, chat_template_kwargs)
                if use_chat_template else prompt)
    overhead = len(tokenizer(
        config.DECODER_PROMPT_TEMPLATE.format(premise="", hypothesis=ex["hypothesis"]),
        add_special_tokens=True)["input_ids"])
    budget = max(32, max_prompt_tokens - overhead - 32)
    prem_ids = tokenizer(ex["premise"], add_special_tokens=False)["input_ids"][:budget]
    premise = tokenizer.decode(prem_ids, skip_special_tokens=True)
    prompt = config.DECODER_PROMPT_TEMPLATE.format(
        premise=premise, hypothesis=ex["hypothesis"])
    return (_wrap_chat_template(prompt, tokenizer, chat_template_kwargs)
            if use_chat_template else prompt)


def normalise_pred(pred: str) -> str:
    """Strip thinking blocks, then return the first matching Hebrew NLI label,
    mapped back to the English label space used in the data."""
    pred = _THINK_RE.sub("", pred).strip()
    if pred in config.HE2EN:
        return config.HE2EN[pred]
    for he, en in config.HE2EN.items():
        if he in pred:
            return en
    return pred  # unmapped — kept as-is so failure modes stay visible


class CausalNLIDataset(Dataset):
    """prompt+target tokenisation with the prompt masked out of the loss.

    If cache_path is given, a previously-tokenised run is loaded from disk
    instead of re-tokenising — a large mixture (all_balanced, 50k+ examples)
    can take minutes, and that cost was otherwise paid again on every restart
    after a 4h SLURM wall-clock kill. Cache key (embedded in cache_path by the
    caller) must include everything that changes the output: model/tokenizer,
    train_spec, seed, max_length."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int,
                cache_path=None, use_chat_template: bool = False,
                chat_template_kwargs: dict | None = None):
        if cache_path is not None and cache_path.exists():
            self._data = torch.load(cache_path, weights_only=True)
            return

        self._data = []
        n_truncated = 0
        for ex in examples:
            target = " " + config.EN2HE[ex["label"]]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [tokenizer.eos_token_id]

            # premise-side truncation: hypothesis + answer cue always survive
            budget = max_length - len(target_ids)
            prompt = build_prompt(ex, tokenizer, budget, use_chat_template, chat_template_kwargs)
            if prompt != build_prompt(ex, tokenizer, None, use_chat_template, chat_template_kwargs):
                n_truncated += 1
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"][:budget]

            input_ids = prompt_ids + target_ids
            labels    = [-100] * len(prompt_ids) + target_ids
            self._data.append({
                "input_ids"     : input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels"        : labels,
            })
        if n_truncated:
            print(f"[decoder-data] truncated {n_truncated}/{len(examples)} prompts "
                  f"to fit max_length={max_length}")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # pid-suffixed: under a DDP launch every rank tokenises
            # independently (deterministic, same output) and would otherwise
            # all write the SAME ".tmp" path concurrently — two processes
            # opening one path for writing at once can interleave/corrupt it.
            # A per-pid tmp name + atomic rename means whichever rank renames
            # last just harmlessly "wins" with identical content.
            tmp = cache_path.with_suffix(f".tmp.{os.getpid()}")
            torch.save(self._data, tmp)
            tmp.rename(cache_path)   # atomic — never leaves a half-written cache file

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class _CausalCollator:
    """Right-pad input_ids/attention_mask with pad_token and labels with -100."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            pad = max_len - len(item["input_ids"])
            out["input_ids"].append(item["input_ids"] + [self.pad_id] * pad)
            out["attention_mask"].append(item["attention_mask"] + [0] * pad)
            out["labels"].append(item["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class LoggingCallback(transformers.TrainerCallback):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.logger.info("step=%d | %s", state.global_step, logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.logger.info("EVAL step=%d | %s", state.global_step, metrics)


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: list[dict],
                         max_length: int, gen_batch_size: int,
                         max_new_tokens: int, log: logging.Logger,
                         use_chat_template: bool = False,
                         chat_template_kwargs: dict | None = None) -> list[str]:
    """Batched greedy generation; returns normalised English label predictions."""
    model.eval()
    tokenizer.padding_side = "left"   # required for batched causal generation
    device = next(model.parameters()).device
    preds: list[str] = []
    prompt_budget = max_length - max_new_tokens
    for i in range(0, len(examples), gen_batch_size):
        batch = examples[i:i + gen_batch_size]
        # premise-side truncation (same as training); tokenizer max_length is
        # only a final guard and should virtually never trigger
        prompts = [build_prompt(ex, tokenizer, prompt_budget, use_chat_template, chat_template_kwargs)
                  for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=prompt_budget
                        ).to(device)
        # some tokenizers (e.g. DictaLM's) emit token_type_ids even for a
        # single sequence; decoder .generate() rejects it as an unknown kwarg
        enc.pop("token_type_ids", None)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            preds.append(normalise_pred(text))
        if (i // gen_batch_size) % 20 == 0:
            log.info("  generated %d/%d", min(i + gen_batch_size, len(examples)),
                     len(examples))
    tokenizer.padding_side = "right"
    return preds


USE_VLLM_EVAL = bool(os.environ.get("USE_VLLM_EVAL"))  # opt-in; default off,
                                                         # existing HF-generate path is
                                                         # untouched unless this is set


def _vllm_ok_for_adapter(flags: dict) -> bool:
    """Whether the FINE-TUNED (adapter-applied) eval may go through vLLM.

    vLLM does NOT apply LoRA deltas to MoE *expert* layers — only
    to attention projections. Models whose adapter covers expert FFNs
    (lora_target="all-linear" on an MoE architecture) therefore evaluate as
    ~the base model, silently.

    Caught on Qwen/Qwen3.5-35B-A3B: training is healthy (LoRA attaches,
    22,477,440 params, loss -> 2e-05) but every fine-tuned score matched its
    own zero-shot to within +/-0.001, and only 15-17 of 884 hebnli predictions
    differed from zero-shot (~1.8%) — the residual is the attention-only part
    of the adapter that vLLM *does* apply. The sibling benchmark
    (finetune_heb_nli) fine-tunes the same model successfully precisely
    because its fine-tuned eval never uses vLLM.

    Gated per-model via MODEL_FLAGS["vllm_lora_unsupported"] rather than by
    sniffing the architecture, so adding an MoE model is an explicit decision.
    Zero-shot eval is unaffected (no adapter involved) and stays on vLLM.
    """
    return not flags.get("vllm_lora_unsupported", False)
VLLM_VENV_PY = os.environ.get("VLLM_PYTHON", sys.executable)  # interpreter with vLLM installed
                                                                  # venv — its vLLM/torch
                                                                  # pairing never has to
                                                                  # match the training env
VLLM_WORKER = Path(__file__).resolve().parent / "vllm_worker.py"


def generate_predictions_vllm_batch(model_name: str, tokenizer,
                                    batch_examples: dict[str, list[dict]],
                                    max_length: int, max_new_tokens: int,
                                    log: logging.Logger,
                                    adapter_path: str | None = None,
                                    trust_remote_code: bool = False,
                                    lora_rank: int = 32,
                                    use_chat_template: bool = False,
                                    chat_template_kwargs: dict | None = None,
                                    enforce_eager: bool = False,
                                    disable_flashinfer_autotune: bool = False) -> dict[str, list[str]]:
    """vLLM equivalent of generate_predictions(), batched across every named
    example set (dev + each test set) in ONE subprocess/engine lifetime —
    engine cold-start is real wall-clock cost, so this pays it once per
    experiment rather than once per test set. Runs under a dedicated venv
    (VLLM_VENV_PY) via subprocess so its vLLM/torch versions never have to
    match whatever the calling process's training stack uses."""
    prompt_budget = max_length - max_new_tokens
    prompt_batch = {
        name: [build_prompt(ex, tokenizer, prompt_budget, use_chat_template, chat_template_kwargs)
              for ex in examples]
        for name, examples in batch_examples.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
        batch_file = Path(tmp) / "batch.json"
        out_file = Path(tmp) / "out.json"
        batch_file.write_text(json.dumps(prompt_batch, ensure_ascii=False),
                              encoding="utf-8")
        cmd = [VLLM_VENV_PY, str(VLLM_WORKER),
               "--model", model_name,
               "--adapter-path", adapter_path or "",
               "--batch-file", str(batch_file),
               "--output-file", str(out_file),
               # +256 safety margin (was +64 — proven insufficient in
               # production, see build_prompt()'s matching margin comment):
               # prompts were truncated to fit max_length using the caller's
               # tokenizer, but vLLM's own tokenizer can count more tokens
               # for the same string (fast/slow tokenizer or special-token
               # differences, compounded by the decode/re-encode round-trip
               # in build_prompt's truncation path) — pad the model's
               # context window rather than the sampling budget so a prompt
               # that's already within the intended bound is never rejected.
               # This is just an allocation ceiling (KV-cache reservation),
               # not extra compute, so a generous margin costs little.
               "--max-model-len", str(max_length + 256),
               "--max-new-tokens", str(max_new_tokens),
               "--max-lora-rank", str(lora_rank)]
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if enforce_eager:
            cmd.append("--enforce-eager")
        if disable_flashinfer_autotune:
            cmd.append("--disable-flashinfer-autotune")
        log.info("[vllm] dispatching batch (%s) to worker venv …",
                 ", ".join(f"{k}={len(v)}" for k, v in batch_examples.items()))
        VLLM_WORKER_TIMEOUT_S = 6 * 60 * 60
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=VLLM_WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.error("[vllm] worker exceeded %dh and was killed -- treating as a "
                      "failed eval rather than blocking this unit forever",
                      VLLM_WORKER_TIMEOUT_S // 3600)
            raise RuntimeError(f"vllm_worker timed out after {VLLM_WORKER_TIMEOUT_S}s")
        if r.returncode != 0:
            log.error("[vllm] worker failed (full stderr, %d chars):\n%s", len(r.stderr), r.stderr)
            raise RuntimeError(f"vllm_worker failed (rc={r.returncode})")
        if r.stdout:
            log.info("[vllm] worker output:\n%s", r.stdout.strip())
        raw = json.loads(out_file.read_text(encoding="utf-8"))
    return {name: [normalise_pred(t) for t in texts] for name, texts in raw.items()}


def run_decoder_zero_shot(
    model_name: str,
    log: logging.Logger,
) -> dict:
    """No-training baseline: evaluate the base model on all 8 test sets with
    the same prompt + greedy generation used after fine-tuning. Deterministic,
    so it runs once (no seeds). Results → results/<model>/zero_shot/."""
    hp    = config.DECODER_SFT
    flags = config.MODEL_FLAGS.get(model_name, {})
    gen_bs = flags.get("gen_batch_size", hp["gen_batch_size"])
    out_run  = config.RESULTS_DIR / config.safe_name(model_name) / "zero_shot"
    pred_dir = out_run / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    test_sets = data_utils.load_all_test_sets()

    trust_remote_code = flags.get("trust_remote_code", False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
        **flags.get("tokenizer_kwargs", {}))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_chat_template = flags.get("use_chat_template", False)
    chat_template_kwargs = flags.get("chat_template_kwargs")
    max_new_tokens = flags.get("max_new_tokens", hp["max_new_tokens"])
    enforce_eager = flags.get("enforce_eager", False)
    disable_flashinfer_autotune = flags.get("disable_flashinfer_autotune", False)
    if USE_VLLM_EVAL:
        log.info("[vllm] zero-shot eval via worker venv: %s", model_name)
        all_preds = generate_predictions_vllm_batch(
            model_name, tokenizer, test_sets, hp["max_length"], max_new_tokens,
            log, trust_remote_code=trust_remote_code, use_chat_template=use_chat_template,
            chat_template_kwargs=chat_template_kwargs,
            enforce_eager=enforce_eager, disable_flashinfer_autotune=disable_flashinfer_autotune)
        def _gen(ds_name, examples):
            return all_preds[ds_name]
    else:
        model_kwargs: dict = dict(
            dtype=_autocast_dtype(),
            trust_remote_code=trust_remote_code,
        )
        if flags.get("attn_implementation"):
            model_kwargs["attn_implementation"] = flags["attn_implementation"]
        model_kwargs["device_map"] = os.environ.get("DEVICE_MAP_OVERRIDE") or flags.get("device_map", "cuda:0")
        log.info("Loading model %s for zero-shot eval (%s)", model_name, model_kwargs)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        def _gen(ds_name, examples):
            return generate_predictions(model, tokenizer, examples, hp["max_length"],
                                        gen_bs, max_new_tokens, log,
                                        use_chat_template=use_chat_template,
                                        chat_template_kwargs=chat_template_kwargs)

    test_metrics: dict[str, dict] = {}
    y_true_all: list = []
    y_pred_all: list = []
    for ds_name, examples in test_sets.items():
        log.info("Zero-shot eval on test set: %s (n=%d)", ds_name, len(examples))
        y_pred = _gen(ds_name, examples)
        y_true = [ex["label"] for ex in examples]
        report = metrics_utils.full_report(y_true, y_pred, config.LABELS_EN)
        report["n_bad_preds"] = sum(1 for p in y_pred if p not in config.LABEL2ID)
        test_metrics[ds_name] = report
        log.info("  %s → %s", ds_name, report)
        with open(pred_dir / f"{ds_name}.jsonl", "w", encoding="utf-8") as f:
            for ex, pred in zip(examples, y_pred):
                f.write(json.dumps({"id": ex["id"], "gold": ex["label"],
                                    "prediction": pred}, ensure_ascii=False) + "\n")
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    results = {
        "model"        : model_name,
        "architecture" : "decoder",
        "approach"     : "zero_shot_generation",
        "train_dataset": "zero_shot",
        "prompt_style" : config.DECODER_PROMPT_STYLE,
        "test"         : test_metrics,
        # "together" scores across all evaluated test sets — pooled: one
        # score over the concatenated union; averaged: mean of the per-set
        # scores above. Different statistics, both kept (see metrics_utils.
        # average_reports's docstring for why).
        "test_pooled"  : metrics_utils.full_report(y_true_all, y_pred_all, config.LABELS_EN),
        "test_averaged": metrics_utils.average_reports(test_metrics),
    }
    with open(out_run / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info("Zero-shot metrics written → %s", out_run / "metrics.json")
    return results


def run_decoder_experiment(
    model_name: str,
    train_spec: str,
    log: logging.Logger,
    seed: int = config.DEFAULT_SEED,
    max_per_dataset: int | None = None,
    keep_checkpoint: bool = False,
) -> dict:
    """LoRA fine-tune *model_name* on *train_spec*, evaluate on all 8 test sets."""
    import peft

    hp    = config.DECODER_SFT
    flags = config.MODEL_FLAGS.get(model_name, {})
    gen_bs = flags.get("gen_batch_size", hp["gen_batch_size"])
    out_run  = config.run_dir(model_name, train_spec, seed)
    out_ckpt = config.ckpt_dir(model_name, train_spec, seed)
    pred_dir = out_run / "predictions"
    for d in (out_run, out_ckpt, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── data ─────────────────────────────────────────────────────────────────
    train_data, dev_data = data_utils.load_train_dev(train_spec, seed, max_per_dataset)
    test_sets = data_utils.load_all_test_sets()

    _cap = getattr(config, "EVAL_DEV_CAP", None)
    if _cap and len(dev_data) > _cap:
        import collections as _c, random as _r
        _by = _c.defaultdict(list)
        for _ex in dev_data:
            _by[_ex["label"]].append(_ex)
        _rng = _r.Random(seed)
        _per = max(1, _cap // max(1, len(_by)))
        _samp = []
        for _lab in sorted(_by):
            _rows = sorted(_by[_lab], key=lambda r: (r["premise"], r["hypothesis"]))
            _rng.shuffle(_rows)
            _samp += _rows[:_per]
        _rng.shuffle(_samp)
        log.info("EVAL_DEV_CAP: dev %d -> %d (label-stratified, seed=%d) -- "
                 "periodic eval only; test metrics use the full test sets",
                 len(dev_data), len(_samp), seed)
        dev_data = _samp

    log.info("train_spec=%s  train=%d  dev=%d  test sets: %s",
             train_spec, len(train_data), len(dev_data),
             {k: len(v) for k, v in test_sets.items()})

    # ── tokenizer ────────────────────────────────────────────────────────────
    trust_remote_code = flags.get("trust_remote_code", False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
        **flags.get("tokenizer_kwargs", {}))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token was None → set to eos_token")

    use_chat_template = flags.get("use_chat_template", False)
    chat_template_kwargs = flags.get("chat_template_kwargs")
    log.info("Tokenising (max_length=%d)", hp["max_length"])
    cache_dir = (config.TOKENIZE_CACHE_DIR / config.safe_name(model_name)
                / f"{train_spec}_seed{seed}_len{hp['max_length']}_cap{max_per_dataset}"
                / (("chat" + ("_" + "_".join(f"{k}{v}" for k, v in
                              sorted(chat_template_kwargs.items()))
                              if chat_template_kwargs else ""))
                   if use_chat_template else "plain"))
    train_ds = CausalNLIDataset(train_data, tokenizer, hp["max_length"],
                                cache_path=cache_dir / "train.pt",
                                use_chat_template=use_chat_template,
                                chat_template_kwargs=chat_template_kwargs)
    dev_ds   = CausalNLIDataset(dev_data,   tokenizer, hp["max_length"],
                                cache_path=cache_dir / "dev.pt",
                                use_chat_template=use_chat_template,
                                chat_template_kwargs=chat_template_kwargs)

    # ── model + LoRA ─────────────────────────────────────────────────────────
    model_kwargs: dict = dict(
        dtype=_autocast_dtype(),
        trust_remote_code=trust_remote_code,
    )
    if flags.get("attn_implementation"):
        model_kwargs["attn_implementation"] = flags["attn_implementation"]
    if IS_DISTRIBUTED:
        log.info("Distributed launch (WORLD_SIZE=%d) — skipping device_map, "
                 "Trainer will place the model on this rank's own GPU (local_rank=%d)",
                 WORLD_SIZE, LOCAL_RANK)
    else:
        model_kwargs["device_map"] = os.environ.get("DEVICE_MAP_OVERRIDE") or flags.get("device_map", "cuda:0")

    log.info("Loading model %s (%s)", model_name, model_kwargs)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # required for gradient checkpointing to work through a frozen base model:
    # without this the embedding output has requires_grad=False and
    # checkpoint-recomputed backward has no graph to attach to (harmless if
    # checkpointing ends up disabled for this model).
    model.enable_input_require_grads()

    _init_fn = getattr(config, "INIT_FROM", None)
    _init_path = _init_fn(model_name, train_spec, seed) if callable(_init_fn) else None
    if _init_path:
        from pathlib import Path as _P
        if not (_P(_init_path) / "adapter_model.safetensors").exists() and \
           not (_P(_init_path) / "adapter_model.bin").exists():
            raise FileNotFoundError(f"INIT_FROM adapter missing: {_init_path}")
        log.info("INIT_FROM: continuing existing adapter %s", _init_path)
        model = peft.PeftModel.from_pretrained(model, str(_init_path), is_trainable=True)
    else:
        lora_config = peft.LoraConfig(
            r=hp["lora_r"],
            lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
            # per-model override: see MODEL_FLAGS["CohereForAI/aya-expanse-32b"]
            target_modules=flags.get("lora_target", hp["lora_target"]),
            task_type=peft.TaskType.CAUSAL_LM,
        )
        model = peft.get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info("LoRA attached — trainable %s / %s (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    if _init_path and trainable == 0:
        raise RuntimeError("INIT_FROM adapter loaded with 0 trainable params -- "
                           "would silently train nothing")

    # ── training setup ───────────────────────────────────────────────────────
    batch_size = flags.get("batch_size", hp["batch_size"])
    grad_accum = flags.get("grad_accum", hp["grad_accum"])
    eval_batch_size = flags.get("eval_batch_size", hp["eval_batch_size"])
    max_steps = config.compute_max_steps(
        len(train_data), batch_size, grad_accum,
        hp["max_epochs"], hp["max_steps_cap"], hp["min_steps"])
    eval_steps = config.compute_eval_steps(
        len(train_data), batch_size, grad_accum, config.eval_per_epoch_for(train_spec, hp),
        min_eval_steps=hp.get("min_eval_steps", 50))
    optim = flags.get("optim", hp["optim"])
    # a checkpoint's optimizer.pt is tied to the optimizer it was saved under
    # (bitsandbytes' paged/8bit optimizers use "state1"/"state2" keys, plain
    # torch optimizers use "exp_avg"/"exp_avg_sq") — loading the wrong format
    # crashes with KeyError inside bitsandbytes. If a prior run of this exact
    # unit checkpointed under a different optimizer (e.g. before optim was
    # changed in config), keep using that optimizer for this resume instead
    # of silently forcing the current default onto an incompatible checkpoint.
    _existing_ckpt = ckpt_utils.last_complete_checkpoint(out_ckpt, log)
    if _existing_ckpt and "8bit" in optim:
        _opt_state = torch.load(
            Path(_existing_ckpt) / "optimizer.pt", map_location="cpu", weights_only=False
        )["state"]
        if _opt_state and "exp_avg" in next(iter(_opt_state.values())):
            log.warning(
                "checkpoint %s was saved under adamw_torch (pre-optim-switch) — "
                "using adamw_torch for this resume instead of %s to avoid a "
                "bitsandbytes state-format crash", _existing_ckpt, optim)
            optim = "adamw_torch"
    log.info("batch_size=%d  grad_accum=%d  eval_batch_size=%d  max_steps=%d  eval_steps=%d  optim=%s",
             batch_size, grad_accum, eval_batch_size, max_steps, eval_steps, optim)

    training_args = transformers.TrainingArguments(
        output_dir                 = str(out_ckpt),
        max_steps                  = max_steps,
        per_device_train_batch_size= batch_size,
        gradient_accumulation_steps= grad_accum,
        per_device_eval_batch_size = eval_batch_size,
        learning_rate              = hp["lr"],
        warmup_ratio               = hp["warmup_ratio"],
        weight_decay               = hp["weight_decay"],
        optim                      = optim,
        torch_compile              = flags.get("torch_compile", False),
        eval_strategy              = "steps",
        eval_steps                 = eval_steps,
        save_strategy              = "steps",
        save_steps                 = eval_steps,
        save_total_limit           = hp["save_total_limit"],
        load_best_model_at_end     = True,
        metric_for_best_model      = "loss",     # dev loss — generation eval is too slow mid-training
        greater_is_better          = False,
        logging_steps              = 50,
        report_to                  = [],
        seed                       = seed,
        data_seed                  = seed,
        bf16                       = _autocast_dtype() is torch.bfloat16,
        fp16                       = _autocast_dtype() is torch.float16,
        gradient_checkpointing     = flags.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm              = flags.get("max_grad_norm", 1.0),
        label_names                = ["labels"],
    )

    trainer = transformers.Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = dev_ds,
        data_collator = _CausalCollator(tokenizer),
        callbacks     = [
            LoggingCallback(log),
            transformers.EarlyStoppingCallback(
                early_stopping_patience=hp["early_stop_patience"]),
        ],
    )

    # resume mid-training if a previous run of this exact (model, spec, seed)
    # crashed — HF restores the LoRA weights, optimizer and step counter
    last_ckpt = ckpt_utils.last_complete_checkpoint(out_ckpt, log)
    if last_ckpt:
        log.info("Resuming training from checkpoint: %s", last_ckpt)
    log.info("Starting LoRA training …")
    trainer.train(resume_from_checkpoint=last_ckpt)
    log.info("Training done. best_checkpoint=%s dev_loss=%.4f",
             trainer.state.best_model_checkpoint, trainer.state.best_metric or 0.0)

    # full loss curve for later analysis — rank-0 only, see is_world_process_zero
    # gate below (every rank has an identical trainer.state after DDP training)
    if trainer.is_world_process_zero():
        with open(out_run / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # collective — every rank must call this (DDP eval is synchronized), even
    # though only rank 0 goes on to use dev_loss/best_ckpt below
    dev_loss = trainer.evaluate(eval_dataset=dev_ds).get("eval_loss")
    best_ckpt = trainer.state.best_model_checkpoint

    if IS_DISTRIBUTED and not trainer.is_world_process_zero():
        log.info("rank %d: training/eval done — skipping post-training generation "
                 "(rank 0 only), waiting at barrier", LOCAL_RANK)
        if tdist.is_initialized():
            tdist.barrier()
        return {"model": model_name, "train_dataset": train_spec, "seed": seed,
                "skipped_non_main_rank": True}

    # informational only (selection used dev loss) — cap for speed on big dev sets
    dev_gen = dev_data[:hp.get("dev_gen_cap") or len(dev_data)]

    # resume: figure out which test sets already have cached predictions,
    # regardless of which generation path is used below
    cached_preds: dict[str, list[str]] = {}
    need_gen: dict[str, list[dict]] = {}
    for ds_name, examples in test_sets.items():
        cached = data_utils.load_cached_predictions(pred_dir / f"{ds_name}.jsonl", examples)
        if cached is not None:
            cached_preds[ds_name] = cached
        else:
            need_gen[ds_name] = examples

    if USE_VLLM_EVAL and not _vllm_ok_for_adapter(flags):
        log.warning("[vllm] SKIPPING vLLM for fine-tuned eval of %s — "
                    "MODEL_FLAGS['vllm_lora_unsupported'] is set (vLLM does not "
                    "apply LoRA to MoE expert layers; would silently evaluate "
                    "~the base model). Falling back to HF generate.", model_name)

    if USE_VLLM_EVAL and _vllm_ok_for_adapter(flags):
        # release the HF model/trainer from GPU memory before vLLM claims it —
        # the two can't share the device at once
        log.info("[vllm] releasing HF model, dispatching batched eval (adapter=%s)",
                 best_ckpt)
        del trainer
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Generating (vllm) on dev + %d test set(s) …", len(need_gen))
        batch_preds = generate_predictions_vllm_batch(
            model_name, tokenizer, {"dev": dev_gen, **need_gen},
            hp["max_length"], hp["max_new_tokens"], log,
            adapter_path=best_ckpt, trust_remote_code=trust_remote_code,
            lora_rank=hp["lora_r"], use_chat_template=use_chat_template,
            enforce_eager=flags.get("enforce_eager", False),
            disable_flashinfer_autotune=flags.get("disable_flashinfer_autotune", False))
        dev_preds = batch_preds.pop("dev")
        gen_preds = batch_preds
    else:
        log.info("Generating on dev for macro-F1 (%d/%d examples) …",
                 len(dev_gen), len(dev_data))
        dev_preds = generate_predictions(model, tokenizer, dev_gen, hp["max_length"],
                                         gen_bs, hp["max_new_tokens"], log)
        gen_preds = {}
        for ds_name, examples in need_gen.items():
            log.info("Evaluating on test set: %s (n=%d)", ds_name, len(examples))
            gen_preds[ds_name] = generate_predictions(model, tokenizer, examples,
                                                       hp["max_length"], gen_bs,
                                                       hp["max_new_tokens"], log)

    dev_report = metrics_utils.full_report(
        [ex["label"] for ex in dev_gen], dev_preds, config.LABELS_EN)
    dev_report["n_dev_generated"] = len(dev_gen)
    log.info("Dev: %s", dev_report)

    # ── evaluate on all 8 test sets ──────────────────────────────────────────
    test_metrics: dict[str, dict] = {}
    y_true_all: list = []
    y_pred_all: list = []
    for ds_name, examples in test_sets.items():
        y_true = [ex["label"] for ex in examples]

        if ds_name in cached_preds:
            log.info("Test set %s: reusing cached predictions (%d) — eval resume",
                     ds_name, len(cached_preds[ds_name]))
            y_pred = cached_preds[ds_name]
        else:
            y_pred = gen_preds[ds_name]
            pred_file = pred_dir / f"{ds_name}.jsonl"
            with open(pred_file, "w", encoding="utf-8") as f:
                for ex, pred in zip(examples, y_pred):
                    f.write(json.dumps({"id": ex["id"], "gold": ex["label"],
                                        "prediction": pred}, ensure_ascii=False) + "\n")

        n_bad  = sum(1 for p in y_pred if p not in config.LABEL2ID)
        report = metrics_utils.full_report(y_true, y_pred, config.LABELS_EN)
        report["n_bad_preds"] = n_bad
        if n_bad:
            log.warning("  %d/%d predictions could not be mapped to a label",
                        n_bad, len(y_pred))
        test_metrics[ds_name] = report
        log.info("  %s → %s", ds_name, report)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    # ── save metrics + adapter ───────────────────────────────────────────────
    results = {
        "model"        : model_name,
        "architecture" : "decoder",
        "approach"     : "lora_sft_generation",
        "train_dataset": train_spec,
        "seed"         : seed,
        "n_train"      : len(train_data),
        "n_dev"        : len(dev_data),
        "max_steps"    : max_steps,
        "max_length"   : hp["max_length"],
        "prompt_style" : config.DECODER_PROMPT_STYLE,
        "hyperparams"  : dict(hp),
        "model_flags"  : dict(flags),
        "resumed_from" : last_ckpt,
        "finished_at"  : datetime.datetime.now().isoformat(timespec="seconds"),
        "versions"     : {"transformers": transformers.__version__,
                          "torch": torch.__version__},
        "lora"         : {k: hp[k] for k in ("lora_r", "lora_alpha", "lora_dropout")},
        "dev"          : {**dev_report, "loss": round(dev_loss, 4) if dev_loss else None},
        "test"         : test_metrics,
        # "together" scores across all evaluated test sets — pooled: one
        # score over the concatenated union; averaged: mean of the per-set
        # scores above. Different statistics, both kept (see metrics_utils.
        # average_reports's docstring for why).
        "test_pooled"  : metrics_utils.full_report(y_true_all, y_pred_all, config.LABELS_EN),
        "test_averaged": metrics_utils.average_reports(test_metrics),
    }
    with open(out_run / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log.info("Metrics written → %s", out_run / "metrics.json")

    if keep_checkpoint:
        best_dir = out_ckpt / "best_adapter"
        # Must use the SAME condition as the eval branch above: `model` is only
        # deleted when the vLLM path actually ran. On the vllm_lora_unsupported
        # fallback the model is still live, so save_pretrained() is correct.
        if USE_VLLM_EVAL and _vllm_ok_for_adapter(flags):
            # model was released for vLLM — best_ckpt already holds the PEFT
            # adapter saved by Trainer's own checkpointing
            shutil.copytree(best_ckpt, best_dir, dirs_exist_ok=True)
        else:
            model.save_pretrained(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        log.info("LoRA adapter saved → %s", best_dir)
    else:
        for sub in out_ckpt.glob("checkpoint-*"):
            shutil.rmtree(sub, ignore_errors=True)
        log.info("Intermediate checkpoints removed (pass --keep-checkpoints to keep)")

    # release the ranks that are waiting at the barrier above, in case this
    # process's queue has more units after this one
    if IS_DISTRIBUTED and tdist.is_initialized():
        tdist.barrier()

    return results
