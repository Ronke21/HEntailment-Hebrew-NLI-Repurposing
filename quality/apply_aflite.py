#!/usr/bin/env python3
"""
Materialise the AFLite-filtered data, re-probe it and re-score trained models on it.

Runs after run_aflite.py. Outputs:
  1. data/experiments_aflite/<src>/{train,dev,test}.jsonl (HebNLI, LCHAIM and
     Winograd are linked unchanged);
  2. quality/results_tfidf_aflite.json: the hypothesis-only TF-IDF probe on the
     filtered data, same instrument as before filtering;
  3. quality/grid_rescored_aflite.json: every run's per-source macro-F1
     recomputed on the filtered test subset from its saved per-item
     predictions (predictions are keyed by item id).
"""
import json, pathlib, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score, f1_score
REPO=__import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
DATA=REPO/"data/experiments"
OUT =REPO/"data/experiments_aflite"
FLT =REPO/"quality/aflite"
SRC=["heq","qasem","hesum","hebsummaries","heb_paraphrase"]
LAB=["entailment","contradiction","neutral"]

# 1. materialise
keepsets={}
for s in SRC:
    f=json.load(open(FLT/f"{s}_filter.json")); keep=set(f["keep"]); keepsets[s]=keep
    d=OUT/s; d.mkdir(parents=True,exist_ok=True)
    for split in ("train","dev","test"):
        rows=[json.loads(l) for l in open(DATA/s/f"{split}.jsonl",encoding="utf-8")]
        kept=[r for r in rows if r["id"] in keep]   # rows carry an id; run_aflite keys on it
        with open(d/f"{split}.jsonl","w",encoding="utf-8") as fh:
            for r in kept: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
        print(f"{s}/{split}: {len(kept)}/{len(rows)} kept")
for extra in ("hebnli","lchaim","winograd","hent_all","hebnli_15k","lchaim_max"):
    t=OUT/extra
    if not t.exists(): t.symlink_to(DATA/extra)

probe={}
for s in SRC:
    tr=[json.loads(l) for l in open(OUT/s/"train.jsonl",encoding="utf-8")]
    te=[json.loads(l) for l in open(OUT/s/"test.jsonl",encoding="utf-8")]
    fe=make_union(TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,max_features=50000),
                  TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4),min_df=2,max_features=50000))
    clf=make_pipeline(fe,LogisticRegression(max_iter=2000)).fit(
        [r["hypothesis"] for r in tr],[r["label"] for r in tr])
    pred=clf.predict([r["hypothesis"] for r in te]); gold=[r["label"] for r in te]
    maj=collections.Counter(gold).most_common(1)[0][1]/len(gold)
    probe[s]=dict(acc=round(accuracy_score(gold,pred),4),
                  macro_f1=round(f1_score(gold,pred,average="macro"),4),
                  majority=round(maj,4),n_train=len(tr),n_test=len(te))
    print(f"probe {s}: acc={probe[s]['acc']} (majority {probe[s]['majority']})")
json.dump(probe,open(REPO/"quality/results_tfidf_aflite.json","w"),indent=2)

# 3. re-score the grid on filtered test subsets
import glob
resc={}
for root in ("grid",):
    for pd in glob.glob(str(REPO/f"experiments/{root}/results/*/train_*/seed_*/predictions")):
        unit="/".join(pathlib.Path(pd).parts[-4:-1])
        for s in SRC:
            pf=pathlib.Path(pd)/f"{s}.jsonl"
            if not pf.exists(): continue
            rows=[json.loads(l) for l in open(pf,encoding="utf-8")]
            # prediction rows carry the test row's id (same key as the filter)
            kept=[r for r in rows if r["id"] in keepsets[s]]
            if len(kept)<20: continue
            f1=f1_score([r["gold"] for r in kept],[r["prediction"] for r in kept],
                        labels=LAB,average="macro",zero_division=0)
            resc.setdefault(unit,{})[s]=dict(macro_f1=round(f1,4),n=len(kept),n_full=len(rows))
json.dump(resc,open(REPO/"quality/grid_rescored_aflite.json","w"),indent=2)
print(f"re-scored {len(resc)} units -> grid_rescored_aflite.json")
