#!/usr/bin/env python3
"""
Low-resource (Scenario 3) and neutral-selection ablation report, 3 seeds.
CPU only, reads metrics.json. Writes s3_ablation_report.md next to this file.

Scenario 3: on HebNLI, for each model and each N in {100, 500, 1k, 5k, 15k}:
baseline (HebNLI-N from scratch) vs augmented (initialised from the model's own
hent_all checkpoint, then the same N rows). Mean over seeds 42/43/44, with the
per-seed difference. Baseline at 15k is the main grid's hebnli_15k.

Ablation: xlm-roberta-large trained on each source with distant neutrals
(experiments/ablation) vs similar neutrals (the main grid), scored on all test sets.
"""
import json, statistics as st
from pathlib import Path
E = Path(__file__).resolve().parent
S3, AB, GR = E / "s3" / "results", E / "ablation" / "results", E / "grid" / "results"
SEEDS = (42, 43, 44)
MODELS = ["dicta-il/neodictabert", "FacebookAI/xlm-roberta-large", "dicta-il/DictaLM-3.0-24B-Base", "google/gemma-4-31B-it"]
NS = ["100", "500", "1k", "5k", "15k"]
SRC = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
sh = lambda m: m.split("/")[1]

def get(root, m, spec, seed, ts):
    p = root / m.replace("/", "__") / f"train_{spec}" / f"seed_{seed}" / "metrics.json"
    return json.load(open(p))["test"][ts]["macro_f1"] if p.exists() else None
def zs(root, m, ts):
    p = root / m.replace("/", "__") / "zero_shot" / "metrics.json"
    return json.load(open(p))["test"][ts]["macro_f1"] if p.exists() else None

L = ["# Scenario 3 and ablation report", "", "Macro-F1, mean over seeds 42/43/44; ± is the seed SD.", "",
     "## Scenario 3: low-resource HebNLI (scored on HebNLI)", "",
     "| model | N | baseline | augmented | gain | seeds with gain > 0 |", "|---|---|---|---|---|---|"]
summary = {}
for m in MODELS:
    for n in NS:
        base_root = GR if n == "15k" else S3
        base_spec = "hebnli_15k" if n == "15k" else f"hebnli_{n}"
        b = [get(base_root, m, base_spec, s, "hebnli") for s in SEEDS]
        a = [get(S3, m, f"aug_hebnli_{n}", s, "hebnli") for s in SEEDS]
        pairs = [(x, y) for x, y in zip(b, a) if x is not None and y is not None]
        if not pairs:
            continue
        d = [y - x for x, y in pairs]
        sdv = lambda v: st.stdev(v) if len(v) > 1 else 0.0
        bm, am = st.mean(x for x, _ in pairs), st.mean(y for _, y in pairs)
        L.append(f"| {sh(m)} | {n} | {bm:.3f} ± {sdv([x for x, _ in pairs]):.3f} | {am:.3f} ± {sdv([y for _, y in pairs]):.3f} "
                 f"| {st.mean(d):+.3f} | {sum(x > 0 for x in d)}/{len(d)} |")
        summary[(m, n)] = (st.mean(d), sum(x > 0 for x in d), len(d))
L += ["", "Gain = augmented - baseline, per seed then averaged. Decoder rows at N = 15k compare against the main grid's hebnli_15k.", ""]
L += ["### Gain by N (mean over seeds)", "", "| model | " + " | ".join(NS) + " |", "|---" * (len(NS) + 1) + "|"]
for m in MODELS:
    L.append(f"| {sh(m)} | " + " | ".join(f"{summary[(m, n)][0]:+.3f}" if (m, n) in summary else "–" for n in NS) + " |")
L += ["", "### Hent_all alone (no HebNLI rows) on HebNLI, for reference", "", "| model | hent_all → HebNLI | zero-shot |", "|---|---|---|"]
for m in MODELS:
    v = [get(S3, m, "hent_all", s, "hebnli") for s in SEEDS] if m in ("dicta-il/neodictabert", "FacebookAI/xlm-roberta-large") else [get(GR, m, "hent_all", s, "hebnli") for s in SEEDS]
    v = [x for x in v if x is not None]; z = zs(GR, m, "hebnli")
    L.append(f"| {sh(m)} | {st.mean(v):.3f} | {z:.3f} |" if v and z is not None else f"| {sh(m)} | {st.mean(v):.3f} | – |" if v else f"| {sh(m)} | – | – |")

M = "FacebookAI/xlm-roberta-large"
TS = SRC + ["hebnli", "lchaim"]
L += ["", "## Ablation: distant vs similar neutrals (xlm-roberta-large)", "",
      "Train on each source with distant neutrals (ablation) or similar neutrals (main grid). Cells: distant, similar, difference (distant - similar). Own-source test set in **bold**.", "",
      "| trained on | " + " | ".join(TS) + " |", "|---" * (len(TS) + 1) + "|"]
diffs_own, diffs_all = [], []
for s in SRC:
    cells = []
    for t in TS:
        d = [get(AB, M, s, x, t) for x in SEEDS]; si = [get(GR, M, s, x, t) for x in SEEDS]
        pr = [(a, b) for a, b in zip(d, si) if a is not None and b is not None]
        if not pr:
            cells.append("–"); continue
        dm, sm = st.mean(a for a, _ in pr), st.mean(b for _, b in pr)
        c = f"{dm:.3f} / {sm:.3f} / {dm - sm:+.3f}"
        (diffs_own if t == s else diffs_all).append(dm - sm)
        cells.append(f"**{c}**" if t == s else c)
    L.append(f"| {s} | " + " | ".join(cells) + " |")
if diffs_own:
    L += ["", f"Own-source difference (distant - similar): mean {st.mean(diffs_own):+.3f}, range {min(diffs_own):+.3f} to {max(diffs_own):+.3f} over {len(diffs_own)} sources.",
          f"Other test sets: mean {st.mean(diffs_all):+.3f}, range {min(diffs_all):+.3f} to {max(diffs_all):+.3f} over {len(diffs_all)} cells."]
L += ["", "## Scenario 3, per seed and without collapsed baselines", "",
      "| model | N | per seed (baseline → augmented) | gain, collapsed baselines excluded | seeds used |", "|---|---|---|---|---|"]
for m in MODELS:
    for n in NS:
        base_root = GR if n == "15k" else S3
        bs = [get(base_root, m, "hebnli_15k" if n == "15k" else f"hebnli_{n}", x, "hebnli") for x in SEEDS]
        as_ = [get(S3, m, f"aug_hebnli_{n}", x, "hebnli") for x in SEEDS]
        if any(v is None for v in bs + as_):
            continue
        ok = [not (b < 0.5 and max(bs) >= 0.7) for b in bs]
        d = [a - b for a, b, k in zip(as_, bs, ok) if k]
        L.append(f"| {sh(m)} | {n} | " + "; ".join(f"{b:.2f}→{a:.2f}" for b, a in zip(bs, as_)) +
                 f" | {st.mean(d):+.3f} | {sum(ok)}/3 |" + ("" if all(ok) else " (baseline collapse on " + ", ".join(str(x) for x, k in zip(SEEDS, ok) if not k) + ")"))
L += ["", "A collapsed baseline is a baseline that failed to train (HebNLI macro-F1 < 0.5 while another seed of the same cell is >= 0.7). The augmented arm never collapses.", ""]
L += ["## Ablation, own-source test set per seed (distant | similar)", "", "| trained on | seed 42 | seed 43 | seed 44 |", "|---|---|---|---|"]
for src in SRC:
    cs = []
    for x in SEEDS:
        d_, s_ = get(AB, M, src, x, src), get(GR, M, src, x, src)
        cs.append(f"{d_:.3f} | {s_:.3f}" if d_ is not None and s_ is not None else "–")
    L.append(f"| {src} | " + " | ".join(c.replace(" | ", " / ") for c in cs) + " |")
if diffs_own:
    # leave-one-out on the own-source difference
    L += ["", "Own-source difference (distant - similar) per source: " + ", ".join(f"{s_}: {d_:+.3f}" for s_, d_ in zip(SRC, diffs_own)),
          f"Mean without HeSum (distant trainings near chance on two of three seeds): {st.mean(d_ for s_, d_ in zip(SRC, diffs_own) if s_ != 'hesum'):+.3f}."]
(E / "s3_ablation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
