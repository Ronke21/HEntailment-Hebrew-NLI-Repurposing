#!/usr/bin/env python3
"""
Corpus-level probes on the finished corpus (CPU). Writes corpus_probes.json
next to this file.

  label mix   : train/dev/test label shares per source, full corpus and the
                experiments' training sample (data/experiments);
  hyp-only    : TF-IDF hypothesis-only probe on the experiments' sample,
                train -> test (the unfiltered number for apply_aflite.py);
  premise-only: 5-fold CV, TF-IDF word 1-2gram -> LR on the premise alone,
                per source, on the full corpus.
"""
import collections, json, os, pathlib, statistics as st
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, make_union

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
FULL = REPO / "data/corpus"; SAMPLE = REPO / "data/experiments"
SRC = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
LAB = ["entailment", "contradiction", "neutral"]
load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
out = {}

# ---- A7: label mixes
mix = {}
for s in SRC:
    mix[s] = {}
    for tag, root in (("full", FULL), ("sample", SAMPLE)):
        for sp in ("train", "dev", "test"):
            c = collections.Counter(r["label"] for r in load(root / s / f"{sp}.jsonl")); n = sum(c.values())
            mix[s][f"{tag}_{sp}"] = {"n": n, **{l: round(c[l] / n, 3) for l in LAB}}
out["label_mix"] = mix; print("A7 done", flush=True)

# ---- P0: unfiltered hypothesis-only probe on the sample (same code as apply_aflite step 2)
p0 = {}
for s in SRC:
    tr, te = load(SAMPLE / s / "train.jsonl"), load(SAMPLE / s / "test.jsonl")
    fe = make_union(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000),
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000))
    clf = make_pipeline(fe, LogisticRegression(max_iter=2000)).fit([r["hypothesis"] for r in tr], [r["label"] for r in tr])
    pred = clf.predict([r["hypothesis"] for r in te]); gold = [r["label"] for r in te]
    maj = collections.Counter(gold).most_common(1)[0][1] / len(gold)
    p0[s] = {"acc": round(accuracy_score(gold, pred), 4), "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
             "majority": round(maj, 4), "n_train": len(tr), "n_test": len(te)}
    print("P0", s, p0[s], flush=True)
out["P0_hyponly_unfiltered_sample"] = p0

# ---- B4: premise-only on the full corpus
b4 = {}
for s in SRC:
    rows = [r for sp in ("train", "dev", "test") for r in load(FULL / s / f"{sp}.jsonl")]
    X = [r["premise"] for r in rows]; y = np.array([r["label"] for r in rows])
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True),
                        LogisticRegression(max_iter=1000, C=1.0))
    acc = float(cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=1).mean())
    maj = float(max(np.mean(y == l) for l in set(y)))
    b4[s] = {"n": len(rows), "premise_only_acc": round(acc, 4), "majority": round(maj, 4)}
    print("B4", s, b4[s], flush=True)
out["premise_only_full"] = b4

json.dump(out, open(pathlib.Path(__file__).resolve().parent / "corpus_probes.json", "w"), indent=1)
print("DONE", flush=True)
