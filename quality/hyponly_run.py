#!/usr/bin/env python
"""
04_hyponly_run.py -- one hypothesis-only cell (arm x dataset).

Trains a classifier on HYPOTHESES ALONE (premise withheld) from a dataset's
own train split and scores it on that dataset's balanced 300/1500-item
evaluation sample from 03_sample_hyponly_eval.py.

ARMS
  tfidf      TF-IDF word 1-2grams -> logistic regression        (CPU)
  encoder    fine-tuned encoder, 3-class head                   (1 GPU)
             --model dicta-il/neodictabert | intfloat/multilingual-e5-large
  llm        LoRA-fine-tuned decoder, label-word generation     (1 big GPU)
             --model google/gemma-4-31B-it | dicta-il/DictaLM-3.0-24B-Base

WHY THE TRAIN CAP
Every arm trains on a class-balanced 1,900-per-label (5,700) subsample of the
dataset's own train split. LCAHIM's smallest train class has 1,946 examples,
so this is the largest balanced size every one of the eight conditions can
supply. Holding it fixed matters: hypothesis-only accuracy rises with
training size, so unequal train splits (HebNLI has 300k, HebSummaries 10k)
would confound "this corpus has artifacts" with "this corpus is bigger".

Encoder hyper-parameters follow Proebsting and Poliak (COLING 2025): 1 epoch,
AdamW, lr 2e-5, weight decay 0.01, batch 16. The decoder arm reuses the
paper's own LoRA settings (r=16, alpha=32, all-linear) so it is comparable to
the transfer grid.

USAGE
    python 04_hyponly_run.py --arm tfidf   --dataset heq
    python 04_hyponly_run.py --arm encoder --dataset heq --model dicta-il/neodictabert
    python 04_hyponly_run.py --arm llm     --dataset heq --model google/gemma-4-31B-it
"""

import argparse, json, os, random, re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CORPUS = Path(os.environ["HYPONLY_CORPUS"]) if "HYPONLY_CORPUS" in os.environ else REPO_ROOT / "data" / "corpus"
SAMPLES = Path(os.environ["HYPONLY_SAMPLES"]) if "HYPONLY_SAMPLES" in os.environ else HERE / "hyponly_samples"
OUT = Path(os.environ["HYPONLY_OUT"]) if "HYPONLY_OUT" in os.environ else HERE / "hyponly_results"
LABELS = ["entailment", "contradiction", "neutral"]
PER_LABEL_TRAIN = 1900

SOURCES = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
# the pooled condition trains on the union of the five sources' train splits
TRAIN_OF = {d: [d] for d in SOURCES + ["hebnli", "lchaim"]}
TRAIN_OF["hentailment_all"] = SOURCES

HEB = {"entailment": "היסק", "contradiction": "סתירה", "neutral": "ניטרלי"}


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def balanced_train(dataset, seed):
    rows = [r for d in TRAIN_OF[dataset] for r in load_jsonl(CORPUS / d / "train.jsonl")]
    rng = random.Random(seed)
    by = {l: [r for r in rows if r.get("label") == l] for l in LABELS}
    out = []
    for l in LABELS:
        n = min(PER_LABEL_TRAIN, len(by[l]))
        out.extend(rng.sample(by[l], n))
    rng.shuffle(out)
    return out


def score(y_true, y_pred, y_train):
    from sklearn.metrics import accuracy_score, f1_score
    maj = max(set(y_train), key=y_train.count)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        majority_accuracy=float(accuracy_score(y_true, [maj] * len(y_true))),
        n_train=len(y_train), n_eval=len(y_true),
        per_label_f1={l: float(f) for l, f in zip(
            sorted(set(y_true) | set(y_pred)),
            f1_score(y_true, y_pred, average=None,
                     labels=sorted(set(y_true) | set(y_pred)), zero_division=0))},
    )


def run_tfidf(tr, ev, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000)
    X = vec.fit_transform([r["hypothesis"] for r in tr])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X, [r["label"] for r in tr])
    pred = clf.predict(vec.transform([r["hypothesis"] for r in ev]))
    return score([r["label"] for r in ev], list(pred), [r["label"] for r in tr])


def run_encoder(tr, ev, seed, ckpt):
    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, set_seed)
    set_seed(seed)
    l2i = {l: i for i, l in enumerate(LABELS)}
    trc = "dictabert" in ckpt.lower()      # NeoDictaBERT ships custom modelling code
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=trc)

    class DS(Dataset):
        def __init__(self, rows):
            self.e = tok([r["hypothesis"] for r in rows], truncation=True,
                         max_length=256, padding="max_length")
            self.y = [l2i[r["label"]] for r in rows]
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            d = {k: torch.tensor(v[i]) for k, v in self.e.items()}
            d["labels"] = torch.tensor(self.y[i]); return d

    m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=3, trust_remote_code=trc)
    args = TrainingArguments(output_dir=f"/tmp/hyp_{os.getpid()}", num_train_epochs=1,
                             learning_rate=2e-5, weight_decay=0.01,
                             per_device_train_batch_size=16, per_device_eval_batch_size=64,
                             logging_steps=200, save_strategy="no", report_to=[], seed=seed,
                             bf16=torch.cuda.is_available())
    t = Trainer(model=m, args=args, train_dataset=DS(tr))
    t.train()
    pred = [LABELS[i] for i in np.asarray(t.predict(DS(ev)).predictions).argmax(-1)]
    return score([r["label"] for r in ev], pred, [r["label"] for r in tr])


def run_llm(tr, ev, seed, ckpt):
    import torch
    from datasets import Dataset as HFDataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, set_seed)
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def prompt(h):
        return ("סווג את המשפט הבא לאחת משלוש הקטגוריות: היסק, סתירה, ניטרלי.\n"
                f"משפט: {h}\nתשובה:")

    train_ds = HFDataset.from_list(
        [{"text": prompt(r["hypothesis"]) + " " + HEB[r["label"]] + tok.eos_token} for r in tr])
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda:0")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    cfg = SFTConfig(output_dir=f"/tmp/hypllm_{os.getpid()}", num_train_epochs=1,
                    learning_rate=1e-4, per_device_train_batch_size=4,
                    gradient_accumulation_steps=4, logging_steps=50, save_strategy="no",
                    report_to=[], seed=seed, bf16=True, max_length=256,
                    completion_only_loss=False)
    SFTTrainer(model=model, args=cfg, train_dataset=train_ds).train()

    model.eval()
    inv = {v: k for k, v in HEB.items()}
    preds = []
    with torch.no_grad():
        for i in range(0, len(ev), 16):
            batch = [prompt(r["hypothesis"]) for r in ev[i:i + 16]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda:0")
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                txt = tok.decode(out[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                lab = next((inv[h] for h in HEB.values() if h in txt), "neutral")
                preds.append(lab)
    return score([r["label"] for r in ev], preds, [r["label"] for r in tr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["tfidf", "encoder", "llm"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr = balanced_train(a.dataset, a.seed)
    ev = load_jsonl(SAMPLES / f"{a.dataset}.jsonl")
    tag = a.model.split("/")[-1] if a.model else "tfidf_lr"
    print(f"[{a.dataset}] arm={a.arm} model={tag} train={len(tr)} eval={len(ev)}", flush=True)

    res = (run_tfidf(tr, ev, a.seed) if a.arm == "tfidf" else
           run_encoder(tr, ev, a.seed, a.model) if a.arm == "encoder" else
           run_llm(tr, ev, a.seed, a.model))
    res.update(dataset=a.dataset, arm=a.arm, model=a.model or "tfidf+LR", seed=a.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.dataset}__{a.arm}__{re.sub(r'[^A-Za-z0-9.-]', '_', tag)}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("accuracy", "macro_f1", "majority_accuracy")},
                     indent=None), flush=True)


if __name__ == "__main__":
    main()
