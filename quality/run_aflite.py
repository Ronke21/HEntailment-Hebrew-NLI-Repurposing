#!/usr/bin/env python3
"""
AFLite-style adversarial filtering on the hypothesis-only axis (robustness check).

AFLite (Le Bras et al. 2020) removes items whose label an ensemble of weak
hypothesis-only probes predicts reliably. Trained models are then re-scored on
the filtered test subsets from their saved per-item predictions; nothing is
retrained.

Algorithm (per source, train+dev+test pooled; every item is scored only when
held out):
  repeat R rounds:
    for E ensemble members: sample 80% of items, fit LogisticRegression on
      mE5 hypothesis embeddings -> predict on the held-out 20%
    predictability(item) = share of held-out appearances whose gold label was
      predicted correctly
    remove items with predictability >= TAU (capped at CUT_FRAC per round)
  until fewer than MIN_REMOVE items are removed in a round.
"""
import argparse, json, pathlib, random, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
DATA = REPO/"data/experiments"
SOURCES = ["heq","qasem","hesum","hebsummaries","heb_paraphrase"]
LABELS = ["entailment","contradiction","neutral"]
TAU, ROUNDS, ENSEMBLE, CUT_FRAC, MIN_REMOVE = 0.75, 8, 32, 0.10, 20

def embed(model, tok, texts, dev, bs=64):
    from torch.nn.functional import normalize
    vecs=[]
    for i in range(0,len(texts),bs):
        with torch.no_grad():
            e=tok([f"query: {t}" for t in texts[i:i+bs]],padding=True,truncation=True,
                  max_length=512,return_tensors="pt").to(dev)
            o=model(**e).last_hidden_state; m=e["attention_mask"].unsqueeze(-1)
            vecs.append(normalize((o*m).sum(1)/m.sum(1),dim=-1).cpu().numpy())
    return np.vstack(vecs)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--outdir",required=True)
    a=ap.parse_args(); out=pathlib.Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModel, AutoTokenizer
    dev="cuda" if torch.cuda.is_available() else "cpu"
    enc=AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(dev).eval()
    tok=AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    rng=np.random.RandomState(42)
    summary={}
    for src in SOURCES:
        rows=[]
        for split in ("train","dev","test"):
            for j,l in enumerate(open(DATA/src/f"{split}.jsonl",encoding="utf-8")):
                r=json.loads(l); r["_id"]=r.get("id",f"{split}:{j}"); r["_split"]=split; rows.append(r)
        X=embed(enc,tok,[r["hypothesis"] for r in rows],dev)
        y=np.array([LABELS.index(r["label"]) for r in rows])
        alive=np.ones(len(rows),bool); dropped_rounds=[]
        for rnd in range(ROUNDS):
            idx=np.where(alive)[0]
            if len(idx)<200: break
            correct=np.zeros(len(rows)); seen=np.zeros(len(rows))
            for e in range(ENSEMBLE):
                perm=rng.permutation(idx); k=int(.8*len(idx))
                tr,ho=perm[:k],perm[k:]
                clf=LogisticRegression(max_iter=1000).fit(X[tr],y[tr])
                pred=clf.predict(X[ho])
                seen[ho]+=1; correct[ho]+=(pred==y[ho])
            scored=idx[seen[idx]>0]
            pr=correct[scored]/seen[scored]
            cand=scored[pr>=TAU]
            cap=int(CUT_FRAC*len(idx))
            if len(cand)>cap:
                order=np.argsort(-pr[np.isin(scored,cand)]); cand=cand[order[:cap]]
            if len(cand)<MIN_REMOVE:
                dropped_rounds.append(len(cand)); alive[cand]=False; break
            alive[cand]=False; dropped_rounds.append(len(cand))
            print(f"{src} round {rnd}: removed {len(cand)}, alive {alive.sum()}/{len(rows)}",flush=True)
        keep=[rows[i]["_id"] for i in np.where(alive)[0]]
        drop=[rows[i]["_id"] for i in np.where(~alive)[0]]
        per_split={s:dict(total=sum(1 for r in rows if r["_split"]==s),
                          kept=sum(1 for i in np.where(alive)[0] if rows[i]["_split"]==s))
                   for s in ("train","dev","test")}
        json.dump(dict(keep=keep,drop=drop),open(out/f"{src}_filter.json","w"))
        summary[src]=dict(total=len(rows),kept=int(alive.sum()),
                          drop_frac=round(1-alive.sum()/len(rows),4),
                          rounds=dropped_rounds,per_split=per_split)
        print(f"{src}: kept {alive.sum()}/{len(rows)} ({1-alive.sum()/len(rows):.1%} dropped)",flush=True)
    json.dump(summary,open(out/"summary.json","w"),indent=2)
    print("->",out/"summary.json")

if __name__=="__main__": main()
