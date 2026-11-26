#!/usr/bin/env python3
"""
Paper tables from the saved grid results (CPU; reads metrics.json only).
Writes paper_tables.md next to this file.

T1  transfer to HebNLI and LCHAIM by training source (decoders, delta vs own zero-shot)
T2  reverse direction: HebNLI-15k fine-tuned models on the five HEntailment test sets
T3  in-domain (own-source) scores, every model
T4  zero-shot scores, decoders, all 8 test sets
T5  the full 8 x 8 matrix per model, mean +/- SD over seeds
Cells are means over seeds 42/43/44; "±" is the seed SD.
"""
import json, statistics as st
from pathlib import Path
E = Path(__file__).resolve().parent
R = E / "grid" / "results"
OUT = E / "paper_tables.md"
SEEDS = (42, 43, 44)
DEC = [("dicta-il/DictaLM-3.0-1.7B-Instruct", "DictaLM-3.0-1.7B"), ("dicta-il/DictaLM-3.0-24B-Base", "DictaLM-3.0-24B"),
       ("HebArabNlpProject/Hebatron", "Hebatron"), ("google/gemma-4-12B-it", "gemma-4-12B"),
       ("google/gemma-4-31B-it", "gemma-4-31B"), ("Qwen/Qwen3.6-27B", "Qwen3.6-27B")]
ENC = [("onlplab/alephbert-base", "AlephBERT"), ("dicta-il/neodictabert", "NeoDictaBERT"),
       ("FacebookAI/xlm-roberta-large", "XLM-R-large"), ("facebook/xlm-roberta-xl", "XLM-R-XL"),
       ("intfloat/multilingual-e5-large", "mE5-large")]
SRC = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
SRCN = {"heq": "HeQ", "qasem": "QASem", "hesum": "HeSum", "hebsummaries": "HebSummaries", "heb_paraphrase": "Hebrew-Paraphrase",
        "hent_all": "HEntailment (pooled)", "hebnli_15k": "HebNLI-15k", "lchaim_max": "LCHAIM-max"}
TRAIN = SRC + ["hent_all", "hebnli_15k", "lchaim_max"]
TESTN = {"heq": "HeQ", "qasem": "QASem", "hesum": "HeSum", "hebsummaries": "HebSummaries", "heb_paraphrase": "HebPar",
         "winograd": "Winograd", "hebnli": "HebNLI", "lchaim": "LCHAIM"}
TEST = list(TESTN)
OWN = {"hebnli_15k": "hebnli", "lchaim_max": "lchaim", "hent_all": None}

def metrics(m, spec, seed):
    d = R / m.replace("/", "__") / ("zero_shot" if spec == "zs" else f"train_{spec}/seed_{seed}") / "metrics.json"
    return json.load(open(d))["test"] if d.exists() else None
def vals(m, spec, t):
    if spec == "zs":
        x = metrics(m, "zs", None); return [x[t]["macro_f1"]] if x else []
    return [x[t]["macro_f1"] for s in SEEDS if (x := metrics(m, spec, s))]
mean = lambda v: st.mean(v) if v else None
sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
f3 = lambda x: "–" if x is None else f"{x:.3f}"
fd = lambda x: "–" if x is None else f"{x:+.3f}"
fm = lambda v: "–" if not v else (f"{mean(v):.3f} ± {sd(v):.3f}" if len(v) > 1 else f"{mean(v):.3f}")

L = ["# Paper tables from the grid", "",
     "Source: `experiments/paper_tables.py` on `grid/results` (270 units, 3 seeds, all complete). Macro-F1. "
     "Cells are means over seeds 42/43/44; ± is the seed SD. Decoders have a zero-shot score; encoders do not (random head).", ""]

# ---------- T1
L += ["## T1. Transfer to HebNLI and LCHAIM by training source (decoders: delta vs own zero-shot)", "",
      "Mean over seeds first, then over the six decoders and the two benchmarks (12 model-benchmark cells per row). "
      "\"Positive\" counts cells with a positive mean delta; it is descriptive, not a test.", "",
      "| training source | mean Δ | range of cell Δ | positive cells | mean Δ HebNLI | mean Δ LCHAIM |", "|---|---|---|---|---|---|"]
t1 = {}
for spec in SRC + ["hent_all", "hebnli_15k", "lchaim_max"]:
    cells = {}
    for m, _ in DEC:
        for t in ("hebnli", "lchaim"):
            z, v = mean(vals(m, "zs", t)), mean(vals(m, spec, t))
            if z is not None and v is not None:
                cells[(m, t)] = v - z
    if not cells:
        continue
    d = list(cells.values())
    per = {t: mean([x for (m, tt), x in cells.items() if tt == t]) for t in ("hebnli", "lchaim")}
    t1[spec] = (mean(d), per)
    L.append(f"| {SRCN[spec]} | {fd(mean(d))} | {fd(min(d))} to {fd(max(d))} | {sum(x > 0 for x in d)}/{len(d)} | {fd(per['hebnli'])} | {fd(per['lchaim'])} |")
L += ["", "Rows HebNLI-15k and LCHAIM-max are the in-task controls: they train on one of the two benchmarks, so on that benchmark the delta is the in-task gain.", ""]

L += ["### T1b. Per-model delta on HebNLI (mean over seeds, ± seed SD of the fine-tuned score)", "",
      "| model | zero-shot | " + " | ".join(SRCN[s] for s in SRC) + " | pooled |", "|---" * (len(SRC) + 3) + "|"]
for m, n in DEC:
    z = mean(vals(m, "zs", "hebnli"))
    cells = []
    for s in SRC + ["hent_all"]:
        v = vals(m, s, "hebnli")
        cells.append("–" if not v else f"{fd(mean(v) - z)} ± {sd(v):.3f}")
    L.append(f"| {n} | {f3(z)} | " + " | ".join(cells) + " |")
L += ["", "### T1c. Encoders: absolute HebNLI macro-F1 (chance 0.333), mean ± seed SD", "",
      "| model | " + " | ".join(SRCN[s] for s in SRC) + " | pooled | HebNLI-15k (in-task) |", "|---" * (len(SRC) + 3) + "|"]
for m, n in ENC:
    L.append(f"| {n} | " + " | ".join(fm(vals(m, s, "hebnli")) for s in SRC + ["hent_all", "hebnli_15k"]) + " |")
L.append("")

# ---------- T2
L += ["## T2. Reverse direction: HebNLI-15k fine-tuned models on the five HEntailment test sets", "",
      "Decoders: score (delta vs own zero-shot). Encoders: score only. Means over seeds.", "",
      "| model | " + " | ".join(TESTN[s] for s in SRC) + " |", "|---" * (len(SRC) + 1) + "|"]
for m, n in DEC:
    cs = []
    for s in SRC:
        v, z = mean(vals(m, "hebnli_15k", s)), mean(vals(m, "zs", s))
        cs.append("–" if v is None else f"{v:.3f} ({fd(v - z)})")
    L.append(f"| {n} | " + " | ".join(cs) + " |")
for m, n in ENC:
    L.append(f"| {n} | " + " | ".join(f3(mean(vals(m, "hebnli_15k", s))) for s in SRC) + " |")
dz = {s: mean([mean(vals(m, "zs", s)) for m, _ in DEC]) for s in SRC}
dv = {s: mean([mean(vals(m, "hebnli_15k", s)) for m, _ in DEC]) for s in SRC}
ev = {s: mean([mean(vals(m, "hebnli_15k", s)) for m, _ in ENC]) for s in SRC}
L += ["", "| group mean | " + " | ".join(TESTN[s] for s in SRC) + " |", "|---" * (len(SRC) + 1) + "|",
      "| decoders zero-shot | " + " | ".join(f3(dz[s]) for s in SRC) + " |",
      "| decoders HebNLI-15k | " + " | ".join(f3(dv[s]) for s in SRC) + " |",
      "| decoder Δ | " + " | ".join(fd(dv[s] - dz[s]) for s in SRC) + " |",
      "| encoders HebNLI-15k | " + " | ".join(f3(ev[s]) for s in SRC) + " |", ""]
L += ["In-task references (mean over seeds): HebNLI-15k on HebNLI: decoders "
      f"{f3(mean([mean(vals(m, 'hebnli_15k', 'hebnli')) for m, _ in DEC]))}, encoders {f3(mean([mean(vals(m, 'hebnli_15k', 'hebnli')) for m, _ in ENC]))}; "
      f"LCHAIM-max on LCHAIM: decoders {f3(mean([mean(vals(m, 'lchaim_max', 'lchaim')) for m, _ in DEC]))}, encoders {f3(mean([mean(vals(m, 'lchaim_max', 'lchaim')) for m, _ in ENC]))}.", ""]

# ---------- T3
L += ["## T3. In-domain scores (trained and tested on the same source), mean ± seed SD", "",
      "| model | " + " | ".join(SRCN[s] for s in SRC) + " | pooled, mean over 5 test sets | HebNLI-15k → HebNLI | LCHAIM-max → LCHAIM |", "|---" * (len(SRC) + 4) + "|"]
for m, n in DEC + ENC:
    row = [fm(vals(m, s, s)) for s in SRC]
    pooled = [mean(vals(m, "hent_all", s)) for s in SRC]
    row.append(f3(mean([x for x in pooled if x is not None])) if any(x is not None for x in pooled) else "–")
    row += [fm(vals(m, "hebnli_15k", "hebnli")), fm(vals(m, "lchaim_max", "lchaim"))]
    L.append(f"| {n} | " + " | ".join(row) + " |")
L.append("")

# ---------- T4
L += ["## T4. Zero-shot scores, decoders", "", "| model | " + " | ".join(TESTN[t] for t in TEST) + " |", "|---" * (len(TEST) + 1) + "|"]
for m, n in DEC:
    L.append(f"| {n} | " + " | ".join(f3(mean(vals(m, "zs", t))) for t in TEST) + " |")
L.append("")

# ---------- T5
L += ["## T5. Full matrix per model (rows: training condition; columns: test set), mean ± seed SD", ""]
for m, n in DEC + ENC:
    L += [f"### {n}", "", "| train \\ test | " + " | ".join(TESTN[t] for t in TEST) + " |", "|---" * (len(TEST) + 1) + "|"]
    if (m, n) in DEC:
        L.append("| zero-shot | " + " | ".join(f3(mean(vals(m, "zs", t))) for t in TEST) + " |")
    for spec in TRAIN:
        L.append(f"| {SRCN[spec]} | " + " | ".join(fm(vals(m, spec, t)) for t in TEST) + " |")
    L.append("")

OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:60]))
print("... written", OUT, len(L), "lines")
