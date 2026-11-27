#!/usr/bin/env python3
"""
Neutral-selection ablation: build the distant-neutral training variant.

The corpus pairs each premise with the most similar other-premise hypothesis
as its neutral (NeoDictaBERT embeddings, cosine ceiling 0.7). This script re-picks each
group's neutral as the least similar other-premise hypothesis by mE5 cosine,
leaving premises, entailments and contradictions identical. dev/test are the
standard splits, so the two arms differ only in the training-time neutral policy.

Output: data/experiments_distant/<src>/train.jsonl (+ dev/test links).
The similar arm is the main grid's per-source results.
"""
import json, pathlib, numpy as np, torch
REPO=__import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
DATA=REPO/"data/experiments"
OUT =REPO/"data/experiments_distant"
SRC=["heq","qasem","hesum","hebsummaries","heb_paraphrase"]

def embed(model,tok,texts,dev,bs=64):
    from torch.nn.functional import normalize
    v=[]
    for i in range(0,len(texts),bs):
        with torch.no_grad():
            e=tok([f"query: {t}" for t in texts[i:i+bs]],padding=True,truncation=True,
                  max_length=512,return_tensors="pt").to(dev)
            o=model(**e).last_hidden_state; m=e["attention_mask"].unsqueeze(-1)
            v.append(normalize((o*m).sum(1)/m.sum(1),dim=-1).cpu().numpy())
    return np.vstack(v)

from transformers import AutoModel, AutoTokenizer
dev="cuda" if torch.cuda.is_available() else "cpu"
enc=AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(dev).eval()
tok=AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
for s in SRC:
    rows=[json.loads(l) for l in open(DATA/s/"train.jsonl",encoding="utf-8")]
    groups={}
    for j,r in enumerate(rows): groups.setdefault(r["premise"],{})[r["label"]]=(j,r)
    prem=list(groups)
    ent_hyp=[groups[p].get("entailment",(None,{"hypothesis":""}))[1]["hypothesis"] for p in prem]
    E=embed(enc,tok,ent_hyp,dev)
    P=embed(enc,tok,prem,dev)
    sim=P@E.T                      # premise x other-groups' entailment hypotheses
    np.fill_diagonal(sim,np.inf)   # never pick your own group's hypothesis
    pick=np.argmin(sim,axis=1)     # DISTANT: least similar
    out=[]; swapped=0
    for i,p in enumerate(prem):
        g=groups[p]
        for lab in ("entailment","contradiction"):
            if lab in g: out.append(g[lab][1])
        if "neutral" in g:
            nr=dict(g["neutral"][1]); nr["hypothesis"]=ent_hyp[pick[i]]
            nr["neutral_policy"]="distant_mE5_argmin"; out.append(nr); swapped+=1
    d=OUT/s; d.mkdir(parents=True,exist_ok=True)
    with open(d/"train.jsonl","w",encoding="utf-8") as fh:
        for r in out: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    for sp in ("dev","test"):
        t=d/f"{sp}.jsonl"
        if not t.exists(): t.symlink_to(DATA/s/f"{sp}.jsonl")
    print(f"{s}: {len(out)} rows, {swapped} neutrals re-picked (distant)",flush=True)
for ext in ("hebnli", "lchaim", "lchaim_max", "winograd"):
    t = OUT / ext
    if not t.exists():
        t.symlink_to((DATA / ext).resolve())
print("done")
