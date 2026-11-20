#!/usr/bin/env python3
"""
Configuration of the main grid (read by experiments/src/run_experiments.py).

Train on one of TRAIN_SPECS_DEFAULT (8 conditions: each HEntailment source,
all five pooled (hent_all), HebNLI-15k, LCAHIM-max) and evaluate on 8 test sets
(5 HEntailment + Winograd + HebNLI + LCAHIM); 11 models (5 encoders, 6
decoders); seeds 42, 43, 44; plus a zero-shot run per decoder.

Data live in DATA_ROOT/<name>/{train,dev,test}.jsonl, records
{"premise", "hypothesis", "label"} with label in {entailment, contradiction,
neutral}; built by experiments/build_data_root.py. Results, checkpoints and
logs are written under this experiment's directory.
"""

from __future__ import annotations

import math
import pathlib
from typing import Optional

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR  = SCRIPT_DIR.parent
REPO_ROOT    = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[3])
DATA_ROOT    = REPO_ROOT / "data" / "experiments"


LOGS_DIR        = PROJECT_DIR / "logs"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR     = PROJECT_DIR / "results"
TOKENIZE_CACHE_DIR = PROJECT_DIR / "tokenize_cache"

# ── datasets ──────────────────────────────────────────────────────────────────
# The 5 HEntailment datasets used for per-source training (each has
# train/dev/test; train capped at 1000/class = 3000 rows by build_corpus.py,
# dev/test are the FULL original splits of the corpus):
TRAIN_DATASETS: list[str] = [
    "heq",
    "qasem",
    "hesum",
    "hebsummaries",
    "heb_paraphrase",
]

# Kept defined (data_utils.load_train_dev references these two unconditionally)
# but UNUSED as real train specs in this experiment -- "hent_all" below is a
# pre-built physical dataset directory (exactly 5x3000=15000 rows), not this
# ALL_TASKS/ALL_BALANCED runtime-concatenation mechanism the old experiments
# used. Do not add "all"/"all_balanced" to TRAIN_SPECS_DEFAULT.
ALL_TASKS        = "all"
ALL_BALANCED     = "all_balanced"
ALL_BALANCED_CAP = 10_000

TRAIN_SPECS_DEFAULT: list[str] = TRAIN_DATASETS + ["hent_all", "hebnli_15k", "lchaim_max"]

EVAL_DATASETS: list[str] = [
    "heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase",
    "winograd", "hebnli", "lchaim",
]

# Friendly aliases -> canonical folder names under DATA_ROOT.
DATASET_ALIASES: dict[str, str] = {
    "samqa"       : "qasem",
    "hepar"       : "heb_paraphrase",
    "hesummaries" : "hebsummaries",
    "paraphrase"  : "heb_paraphrase",
    "all"         : "hent_all",   # convenience alias, NOT config.ALL_TASKS
}

# ── labels ────────────────────────────────────────────────────────────────────
LABELS_EN = ["entailment", "contradiction", "neutral"]
LABEL2ID  = {l: i for i, l in enumerate(LABELS_EN)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS_EN)}

# Hebrew label words used as decoder targets.
LABELS_HE = ["היסק", "סתירה", "ניטרלי"]
EN2HE     = dict(zip(LABELS_EN, LABELS_HE))
HE2EN     = dict(zip(LABELS_HE, LABELS_EN))

ENCODER_MODELS: list[str] = [
    "dicta-il/neodictabert",
    "onlplab/alephbert-base",
    "FacebookAI/xlm-roberta-large",
    "facebook/xlm-roberta-xl",
    "intfloat/multilingual-e5-large",
]

DECODER_MODELS: list[str] = [
    "dicta-il/DictaLM-3.0-24B-Base",
    "dicta-il/DictaLM-3.0-1.7B-Instruct",
    "HebArabNlpProject/Hebatron",
    "google/gemma-4-31B-it",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.6-27B",
]

DEFAULT_MODELS: list[str] = ENCODER_MODELS + DECODER_MODELS

# Per-model loading flags -- carried over UNCHANGED from finetune_repurposed_data's
# hard-won engineering fixes (OOM/attn-kernel/chat-template/vLLM-LoRA issues,
# each with its own dated incident note) for every model kept in this
# experiment's roster. Entries for excluded models (aya-expanse-32b,
# gemma-3-27b-it, Qwen3.5-35B-A3B, Qwen3-32B) are dropped, not carried over.
MODEL_FLAGS: dict[str, dict] = {
    # ── encoders ──────────────────────────────────────────────────────────────
    "dicta-il/neodictabert": {
        "trust_remote_code"  : True,
    },
    "onlplab/alephbert-base": {
        "attn_implementation": "eager",
        "zero_token_type_ids": True,   # type_vocab_size=1 -> token_type_ids must stay 0
    },
    # FacebookAI/xlm-roberta-large, facebook/xlm-roberta-xl,
    # intfloat/multilingual-e5-large: no special flags needed.
    #
    # Every SLURM/A100 job here exposes exactly one GPU, so device_map="auto"
    # buys nothing and has triggered huge single-shot alloc failures on some
    # environments -- every large decoder below uses the explicit "cuda:0".
    # ── decoders — Hebrew ─────────────────────────────────────────────────────
    "dicta-il/DictaLM-3.0-24B-Base": {
        "attn_implementation": "sdpa",   # flash_attention_2 fails on this env's
                                          # compute capability -- sdpa needs no
                                          # external compiled kernel
        "device_map"         : "cuda:0",
        "gradient_checkpointing": True,
        "batch_size"          : 2,
        "grad_accum"          : 8,       # preserves effective batch size 16
        # Large-vocab loss-computation OOM at the first eval checkpoint under
        # the global eval_batch_size default -- override down, pure throughput
        # knob, doesn't change the fitted model.
        "eval_batch_size"     : 8,
    },
    "dicta-il/DictaLM-3.0-1.7B-Instruct": {
        "attn_implementation": "sdpa",
        "device_map"         : "cuda:0",
        "gen_batch_size"      : 24,
        "batch_size"          : 2,
        "grad_accum"          : 8,
        "gradient_checkpointing": True,
        "eval_batch_size"     : 2,
    },
    "HebArabNlpProject/Hebatron": {
        "attn_implementation": "eager",
        "trust_remote_code"  : True,
        "device_map"         : "cuda:0",
        "batch_size"         : 8,        # ~47B Mamba-MoE hybrid; OOMs on non-tiny
        "grad_accum"         : 4,        # datasets at higher batch. Preserves eff. batch 32
        "gen_batch_size"     : 48,
        "gradient_checkpointing": True,
        "eval_batch_size"     : 8,       # same large-vocab eval-time OOM class as DictaLM-24B
    },
    # ── decoders — multilingual ───────────────────────────────────────────────
    "google/gemma-4-31B-it": {
        "attn_implementation": "sdpa",   # eager OOMs on long sequences (unfused softmax)
        "device_map"         : "cuda:0",
        "gradient_checkpointing": True,
        "batch_size"          : 2,
        "grad_accum"          : 8,
        # Zero-shot needs this model's own (very new) chat template or
        # generation is incoherent multi-script gibberish; SFT bridges the
        # format gap without it, so this is scoped to zero-shot in the
        # pipeline code, not a training-time prompt change.
        "use_chat_template"   : True,
        "eval_batch_size"     : 2,       # large-vocab eval-time OOM, tightened further
    },
    "google/gemma-4-12B-it": {
        "attn_implementation": "sdpa",
        "device_map"         : "cuda:0",
        "gradient_checkpointing": True,
        "batch_size"          : 2,
        "grad_accum"          : 8,
        # Same gemma-4 zero-shot failure as the 31B-it entry above: without
        # the chat template it never emits a label (degenerates into raw-text
        # continuation/repetition). gemma-3-27b-it does NOT need this --
        # confirmed gemma-4-specific.
        "use_chat_template"   : True,
        "eval_batch_size"     : 2,
        "gen_batch_size"      : 8,
    },
    "Qwen/Qwen3.6-27B": {
        "attn_implementation": "sdpa",
        "device_map"         : "cuda:0",
        "gradient_checkpointing": True,
        "batch_size"          : 2,
        "grad_accum"          : 8,
        "eval_batch_size"     : 2,
        "lora_target"         : ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
        "vllm_lora_unsupported": True,
        "gen_batch_size"      : 8,
    },
}

# ── encoder hyperparameters ───────────────────────────────────────────────────
ENCODER_CLF: dict = dict(
    max_length       = 2048,
    batch_size       = 32,
    eval_batch_size  = 64,
    lr               = 2e-5,
    warmup_ratio     = 0.1,
    weight_decay     = 0.01,
    max_epochs       = 20,
    max_steps_cap    = 8000,    # never binds in practice (measured max 4,685 steps)
    min_steps        = 200,
    eval_per_epoch   = 4,
    early_stop_patience = 5,
    save_total_limit = 2,
)

# ── decoder hyperparameters (LoRA SFT) ────────────────────────────────────────
DECODER_SFT: dict = dict(
    max_length       = 2048,
    batch_size       = 8,
    grad_accum       = 2,
    eval_batch_size  = 8,
    gen_batch_size   = 96,
    dev_gen_cap      = 2000,
    max_new_tokens   = 8,
    lr               = 1e-4,
    warmup_ratio     = 0.03,
    weight_decay     = 0.0,
    max_epochs       = 3,
    max_steps_cap    = 8000,
    min_steps        = 100,
    eval_per_epoch   = 4,
    early_stop_patience = 5,
    save_total_limit = 2,
    lora_r           = 16,
    lora_alpha       = 32,
    lora_dropout     = 0.05,
    lora_target      = "all-linear",
    optim            = "paged_adamw_8bit",
)

# ── decoder prompt ────────────────────────────────────────────────────────────
DECODER_PROMPT_STYLE = "instruction"

DECODER_PROMPT_TEMPLATES: dict[str, str] = {
    "instruction": (
        "לפניך שני משפטים. קבע את היחס הלוגי בין משפט 1 למשפט 2 "
        "וענה במילה אחת בלבד: היסק, סתירה או ניטרלי.\n"
        "משפט 1: {premise}\n"
        "משפט 2: {hypothesis}\n"
        "תשובה:"
    ),
    "minimal": "משפט 1: {premise} משפט 2: {hypothesis}\nתשובה:",
}

DECODER_PROMPT_TEMPLATE = DECODER_PROMPT_TEMPLATES[DECODER_PROMPT_STYLE]

DEFAULT_SEED  = 42
DEFAULT_SEEDS = [42, 43]

# ── smoke test mode ───────────────────────────────────────────────────────────
SMOKE_MODE = False

SMOKE_DATA_CAPS = {"train": 64, "dev": 32, "test": 16}

_SMOKE_ENCODER_OVERRIDES = dict(
    max_length=256, batch_size=8, eval_batch_size=16,
    max_epochs=1, max_steps_cap=8, min_steps=8,
    min_eval_steps=2, early_stop_patience=100,
)
_SMOKE_DECODER_OVERRIDES = dict(
    max_length=256, batch_size=2, grad_accum=1, eval_batch_size=4,
    gen_batch_size=8, max_epochs=1, max_steps_cap=8, min_steps=8,
    min_eval_steps=2, early_stop_patience=100,
)


def enable_smoke_mode() -> None:
    """Switch this config module into smoke-test mode (call before any run)."""
    global SMOKE_MODE, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, TOKENIZE_CACHE_DIR
    SMOKE_MODE      = True
    RESULTS_DIR     = PROJECT_DIR / "results_smoke"
    CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints_smoke"
    LOGS_DIR        = PROJECT_DIR / "logs_smoke"
    TOKENIZE_CACHE_DIR = PROJECT_DIR / "tokenize_cache_smoke"
    ENCODER_CLF.update(_SMOKE_ENCODER_OVERRIDES)
    DECODER_SFT.update(_SMOKE_DECODER_OVERRIDES)


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_dataset(name: str) -> str:
    """Map a user-supplied dataset name/alias to its canonical folder name."""
    canonical = DATASET_ALIASES.get(name.strip().lower(), name.strip().lower())
    valid = set(TRAIN_SPECS_DEFAULT) | set(EVAL_DATASETS)
    if canonical not in valid:
        raise ValueError(f"Unknown dataset {name!r} (canonical: {canonical!r}). "
                         f"Valid: {sorted(valid)}")
    return canonical


def safe_name(model_id: str) -> str:
    """'org/model' -> 'org__model' for filesystem paths."""
    return model_id.replace("/", "__")


def run_dir(model_id: str, train_spec: str, seed: int = DEFAULT_SEED) -> pathlib.Path:
    return RESULTS_DIR / safe_name(model_id) / f"train_{train_spec}" / f"seed_{seed}"


def ckpt_dir(model_id: str, train_spec: str, seed: int = DEFAULT_SEED) -> pathlib.Path:
    return CHECKPOINTS_DIR / safe_name(model_id) / f"train_{train_spec}" / f"seed_{seed}"


def is_run_complete(model_id: str, train_spec: str, seed: int = DEFAULT_SEED) -> bool:
    return (run_dir(model_id, train_spec, seed) / "metrics.json").exists()


def compute_max_steps(n_train: int, batch_size: int, grad_accum: int,
                      max_epochs: int, max_steps_cap: int, min_steps: int) -> int:
    steps_per_epoch = max(1, math.ceil(n_train / (batch_size * grad_accum)))
    return max(min_steps, min(max_steps_cap, steps_per_epoch * max_epochs))


def compute_eval_steps(n_train: int, batch_size: int, grad_accum: int,
                       eval_per_epoch: int, min_eval_steps: int = 50) -> int:
    steps_per_epoch = max(1, math.ceil(n_train / (batch_size * grad_accum)))
    return max(min_eval_steps, steps_per_epoch // eval_per_epoch)


EVAL_PER_EPOCH_OVERRIDE: dict[str, int] = {
}


def eval_per_epoch_for(train_spec: str, hp: dict) -> int:
    return EVAL_PER_EPOCH_OVERRIDE.get(train_spec, hp["eval_per_epoch"])

DEFAULT_SEEDS = [42, 43, 44]


