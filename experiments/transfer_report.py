#!/usr/bin/env python3
"""
Transfer report, both directions, every model (CPU; reads metrics.json).
Writes transfer_report.md next to this file.

Direction 1, HEntailment -> translated NLI: a model fine-tuned on one HEntailment
source (or all five pooled), scored on HebNLI and LCHAIM.
  * decoders: delta against the model's own zero-shot score;
  * encoders have no zero-shot (random head), so they are reported against chance
    (0.333) and against the same encoder fine-tuned on HebNLI (hebnli_15k).
Direction 2, HebNLI -> HEntailment: hebnli_15k runs scored on the five test sets.
Model-level test: one-sided Mann-Whitney U (exact), decoders vs encoders, unit =
model, statistic = the model's mean HebNLI macro-F1 over the five sources.
Seeds 42, 43 and 44 are averaged.

Prior bound: per decoder, x = zero-shot HebNLI macro-F1, y = mean over the five
single sources of (fine-tuned HebNLI - zero-shot HebNLI); Spearman rho with an
exact one-sided permutation p (H1: rho < 0).
"""
import itertools, json, os, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent / "grid" / "results"
DEC = ["dicta-il/DictaLM-3.0-1.7B-Instruct", "dicta-il/DictaLM-3.0-24B-Base", "HebArabNlpProject/Hebatron",
       "google/gemma-4-31B-it", "google/gemma-4-12B-it", "Qwen/Qwen3.6-27B"]
ENC = ["onlplab/alephbert-base", "dicta-il/neodictabert", "FacebookAI/xlm-roberta-large",
       "facebook/xlm-roberta-xl", "intfloat/multilingual-e5-large"]
SRC = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
TRAIN = SRC + ["hent_all"]; HELD = ["hebnli", "lchaim"]
SEEDS = (42, 43, 44)
sh = lambda m: m.split("/")[1].replace("-Instruct", "").replace("-it", "")

def test(m, spec, seed):
    d = R / m.replace("/", "__") / ("zero_shot" if spec == "zs" else f"train_{spec}/seed_{seed}") / "metrics.json"
    return json.load(open(d))["test"] if d.exists() else None
def f1(m, spec, t):
    """mean over available seeds; (value, n_seeds)"""
    v = [x[t]["macro_f1"] for s in SEEDS if (x := test(m, spec, s))]
    if spec == "zs":
        z = test(m, "zs", None); v = [z[t]["macro_f1"]] if z else []
    return (st.mean(v), len(v)) if v else (None, 0)
fmt = lambda x: f"{x:.3f}" if x is not None else "–"
def mwu_greater(a, b):
    """exact one-sided P(U >= observed) that a > b"""
    obs = sum((x > y) + 0.5 * (x == y) for x in a for y in b); pool = a + b; n = len(a); cnt = tot = 0
    for idx in itertools.combinations(range(len(pool)), n):
        aa = [pool[i] for i in idx]; bb = [pool[i] for i in range(len(pool)) if i not in idx]
        u = sum((x > y) + 0.5 * (x == y) for x in aa for y in bb); tot += 1; cnt += u >= obs - 1e-12
    return obs, cnt / tot

def sd(m, spec, t):
    v = [x[t]["macro_f1"] for s in SEEDS if (x := test(m, spec, s))]
    return st.stdev(v) if len(v) > 1 else None
def spearman_exact(x, y):
    """Spearman rho and exact one-sided p for rho <= observed (no ties expected)."""
    n = len(x); rk = lambda v: [sorted(v).index(a) + 1 for a in v]
    rx, ry = rk(x), rk(y)
    r = lambda a, b: 1 - 6 * sum((i - j) ** 2 for i, j in zip(a, b)) / (n * (n * n - 1))
    obs = r(rx, ry); perms = list(itertools.permutations(ry))
    return obs, sum(r(rx, q) <= obs + 1e-12 for q in perms) / len(perms)

L = ["# Transfer report", "", f"Macro-F1 on the HEntailment test sets, mean over seeds {', '.join(map(str, SEEDS))}; "
     "a cell marked with a superscript has fewer seeds (its count). Max seed SD per table is given under it.", ""]
for held in HELD:
    L += [f"## HEntailment -> {held.upper()}", "", "### Decoders (delta vs own zero-shot)", "",
          "| model | zero-shot | " + " | ".join(TRAIN) + " | HebNLI-15k (ref) |", "|---" * (len(TRAIN) + 3) + "|"]
    for m in DEC:
        z, _ = f1(m, "zs", held)
        if z is None: continue
        cells = []
        for s in TRAIN + ["hebnli_15k"]:
            v, n = f1(m, s, held); cells.append(f"{v - z:+.3f}" + ("" if n == len(SEEDS) else f"^{n}") if v is not None else "–")
        L.append(f"| {sh(m)} | {z:.3f} | " + " | ".join(cells) + " |")
    dsd = [x for m in DEC for s in TRAIN + ["hebnli_15k"] if (x := sd(m, s, held)) is not None]
    L += ["", f"max seed SD in this table: {max(dsd):.3f}; median {st.median(dsd):.3f}.", "", "### Encoders (absolute; chance 0.333)", "",
          "| model | " + " | ".join(TRAIN) + " | HebNLI-15k (ref) |", "|---" * (len(TRAIN) + 2) + "|"]
    for m in ENC:
        cells = [fmt(f1(m, s, held)[0]) for s in TRAIN + ["hebnli_15k"]]
        L.append(f"| {sh(m)} | " + " | ".join(cells) + " |")
    esd = [x for m in ENC for s in TRAIN + ["hebnli_15k"] if (x := sd(m, s, held)) is not None]
    L += ["", f"max seed SD in this table: {max(esd):.3f}; median {st.median(esd):.3f}.", ""]
L += ["## Model-level test, HebNLI (mean over the five single sources)", ""]
dm = {sh(m): st.mean(v) for m in DEC if len(v := [f1(m, s, "hebnli")[0] for s in SRC if f1(m, s, "hebnli")[0] is not None]) == 5}
em = {sh(m): st.mean(v) for m in ENC if len(v := [f1(m, s, "hebnli")[0] for s in SRC if f1(m, s, "hebnli")[0] is not None]) == 5}
if dm and em:
    u, p = mwu_greater(list(dm.values()), list(em.values()))
    L += [f"decoders ({len(dm)}): " + ", ".join(f"{k} {v:.3f}" for k, v in dm.items()),
          f"encoders ({len(em)}): " + ", ".join(f"{k} {v:.3f}" for k, v in em.items()),
          f"one-sided Mann-Whitney, decoders > encoders: U = {u:.1f} of {len(dm) * len(em)}, exact p = {p:.4f}", ""]
L += ["## Prior bound, HebNLI (decoders)", "",
      "x = zero-shot HebNLI macro-F1; y = mean over the five single sources of (fine-tuned - zero-shot) on HebNLI.", "",
      "| decoder | zero-shot | mean transfer delta |", "|---|---|---|"]
pb = {}
for m in DEC:
    z = f1(m, "zs", "hebnli")[0]; d = [f1(m, s, "hebnli")[0] for s in SRC]
    if z is not None and all(v is not None for v in d):
        pb[sh(m)] = (z, st.mean(d) - z); L.append(f"| {sh(m)} | {z:.3f} | {st.mean(d) - z:+.3f} |")
if len(pb) >= 3:
    rho, p = spearman_exact([v[0] for v in pb.values()], [v[1] for v in pb.values()])
    L += ["", f"Spearman rho = {rho:+.3f}, n = {len(pb)}, exact one-sided p = {p:.4f}"]
    no_q = {k: v for k, v in pb.items() if not k.startswith("Qwen")}
    if len(no_q) != len(pb) and len(no_q) >= 3:
        rho2, p2 = spearman_exact([v[0] for v in no_q.values()], [v[1] for v in no_q.values()])
        L += [f"without Qwen3.6-27B (the 2-seed figure's set): rho = {rho2:+.3f}, n = {len(no_q)}, exact one-sided p = {p2:.4f}"]
L += ["", "## HebNLI -> HEntailment (hebnli_15k units on the HEntailment test sets)", "",
      "| model | " + " | ".join(SRC) + " |", "|---" * (len(SRC) + 1) + "|"]
for m in DEC + ENC:
    if f1(m, "hebnli_15k", "heq")[0] is None: continue
    z = {s: f1(m, "zs", s)[0] for s in SRC}
    cells = []
    for s in SRC:
        v = f1(m, "hebnli_15k", s)[0]
        cells.append(f"{v:.3f}" + (f" ({v - z[s]:+.3f})" if z[s] is not None else ""))
    L.append(f"| {sh(m)} | " + " | ".join(cells) + " |")
L += ["", "Decoders show the delta against their own zero-shot in brackets; encoders have none."]
out = Path(__file__).resolve().parent / "transfer_report.md"
out.write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
