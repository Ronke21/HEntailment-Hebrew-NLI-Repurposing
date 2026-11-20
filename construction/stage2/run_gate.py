#!/usr/bin/env python3
"""
Stage 2f: artifact gate (CPU).

Per rule (and rule pair), the rule's contradictions are probed against their
own entailment twins (same premises, grouped folds), balanced, chance 0.50:
  hyp_only   : TF-IDF word 1-2 + char_wb 2-4 -> LR on the hypothesis alone
  length_only: LR on hypothesis word/char count and premise ratio
  overlap    : LR on Jaccard and hypothesis-in-premise coverage
Corpus level (3-class, chance 0.33): the same probes, plus the twin-edit
distribution and the rule shares. A rule whose contradictions are recoverable
from the hypothesis alone is added to exclude_rules.txt and the source rebuilt.
Output: <dir>/gate.json
"""
from __future__ import annotations
import argparse, json, os
os.environ.setdefault("OMP_NUM_THREADS", "4")
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
V4 = REPO / "data/corpus"
STAGE1 = REPO / "data/stage1"
LAB = {"entailment": 0, "contradiction": 1, "neutral": 2}
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
rng = np.random.RandomState(42)


def load(d):
    rows = []
    for sp in ("train", "dev", "test"):
        f = d / f"{sp}.jsonl"
        if f.exists(): rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    return rows


def feats(rows):
    X = []
    for r in rows:
        p, h = r["premise"].split(), r["hypothesis"].split(); sp, sh = set(p), set(h)
        X.append([len(h), len(r["hypothesis"]), len(h) / max(1, len(p)),
                  len(sp & sh) / max(1, len(sp | sh)), len(sp & sh) / max(1, len(sh))])
    return np.array(X, dtype=float)


def hyp_only(texts, y, groups):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    X = hstack([w.fit_transform(texts), c.fit_transform(texts)]).tocsr()
    return float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv, groups=groups).mean())


def num_only(X, y, groups):
    return float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv, groups=groups).mean())


def probe_set(rows):
    y = np.array([LAB[r["label"]] for r in rows]); X = feats(rows); T = [r["hypothesis"] for r in rows]
    G = np.array([hash(r["premise"]) % (2**31) for r in rows])
    if len(set(y)) < 2 or min(np.bincount(y)[np.bincount(y) > 0]) < 10:
        return None
    return {"n": len(rows), "majority": round(float(np.bincount(y).max() / len(y)), 3),
            "hyp_only": round(hyp_only(T, y, G), 3), "length_only": round(num_only(X[:, :3], y, G), 3),
            "overlap_only": round(num_only(X[:, 3:], y, G), 3)}


def balanced_pair(ent, con):
    n = min(len(ent), len(con))
    return list(rng.choice(ent, n, replace=False)) + list(rng.choice(con, n, replace=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="heq")
    ap.add_argument("--dir", default="", help="gate a saved variant dir instead of data/corpus/<src>")
    args = ap.parse_args()
    v4dir = Path(args.dir) if args.dir else V4 / args.src
    out = {}
    for name, d in (("hentailment", v4dir),):
        rows = load(d)
        if not rows: continue
        ent = [r for r in rows if r["label"] == "entailment"]; con = [r for r in rows if r["label"] == "contradiction"]
        res = {"corpus_3class": probe_set(rows), "ent_vs_con": probe_set(balanced_pair(ent, con))}
        if name == "hentailment":
            by_rule = defaultdict(list)
            for r in con: by_rule["+".join(r["rules"])].append(r)
            # per rule: the rule's contradictions against THEIR OWN entailment twins
            # (same premises), so "contains a number / a name" cannot masquerade as a
            # label cue; grouped folds keep each twin pair on one side of the split
            ent_by_premise = {r["premise"]: r for r in ent}
            def twin(r): 
                if r.get("source_entailment"):
                    return {"premise": r["premise"], "hypothesis": r["source_entailment"], "label": "entailment"}
                return ent_by_premise.get(r["premise"])
            res["per_rule_ent_vs_con"] = {}
            for rule, rc in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
                if len(rc) < 30: continue
                twins = [t for t in (twin(r) for r in rc) if t]
                res["per_rule_ent_vs_con"][rule] = probe_set(twins + rc)
            fam = defaultdict(list)
            for r in con: fam["compound" if len(r["rules"]) > 1 else "single"].append(r)
            res["single_vs_compound_ent_vs_con"] = {k: probe_set(balanced_pair(ent, v)) for k, v in fam.items() if len(v) >= 30}
            res["edit_tokens_hist"] = {str(k): v for k, v in sorted(__import__("collections").Counter(min(r["n_edit_tokens"], 5) for r in con).items())}
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k in ("corpus_3class", "ent_vs_con")}), flush=True)
    json.dump(out, open(v4dir / "gate.json", "w"), indent=1)
    print(json.dumps(out.get("hentailment", {}).get("per_rule_ent_vs_con", {}), indent=1)); print("DONE")


if __name__ == "__main__":
    main()
