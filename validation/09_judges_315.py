#!/usr/bin/env python3
"""
09_judges_315.py -- score every judge on all 315 validation items (camera-ready Table 5,
Appendix E and F).

Judges: the two annotators and the two zero-shot LLMs (cached answers from
06_llm_judge_validation.py; no API calls are made). Each is scored against the
pipeline's label and against annotator 2. Items a judge marked incoherent, or
answered with no parsable label, are left out of that judge's figures.
Label parsing is identical to 06_llm_judge_validation.parse().

Writes human_validation_results/judges_315.json (does not touch validation_report.json).
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = ["entailment", "contradiction", "neutral"]
SOURCES = ["heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"]
LLMS = {"gemini": "cache_google__gemini-3.5-flash-lite.json",
        "gpt": "cache_openai__gpt-5.4-mini.json"}


def parse(text):
    t = (text or "").strip().lower()
    for lab in LABELS:
        if t == lab:
            return lab
    for lab in LABELS:
        if lab in t:
            return lab
    return None


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in LABELS) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def scores(gold, pred):
    n = len(gold)
    f1 = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(n=n, macro_f1=sum(f1.values()) / 3,
                accuracy=sum(g == p for g, p in zip(gold, pred)) / n,
                kappa=kappa(gold, pred), f1=f1)


def main():
    lab = json.loads((HERE / "human_validation_results" / "labels_315_complete.json").read_text())
    ids = sorted(lab)
    judges = {"A1": {i: lab[i]["a1"] for i in ids}, "A2": {i: lab[i]["a2"] for i in ids}}
    for nm, f in LLMS.items():
        cache = json.loads((HERE / "llm_judge_results" / f).read_text())
        judges[nm] = {i: parse(cache.get(i)) for i in ids}

    out = {}
    for nm, J in judges.items():
        res = {}
        for ref in ["pipeline", "A2"]:
            if nm == ref:
                continue
            refl = {i: (lab[i]["pipeline"] if ref == "pipeline" else lab[i]["a2"]) for i in ids}
            keep = [i for i in ids if J[i] in LABELS and refl[i] in LABELS]
            r = scores([refl[i] for i in keep], [J[i] for i in keep])
            r["by_source"] = {}
            for s in SOURCES:
                ks = [i for i in keep if lab[i]["dataset"] == s]
                r["by_source"][s] = dict(n=len(ks), accuracy=sum(J[i] == refl[i] for i in ks) / len(ks))
            r["by_gold_label"] = {}
            for L in LABELS:
                kl = [i for i in keep if refl[i] == L]
                r["by_gold_label"][L] = dict(n=len(kl), accuracy=sum(J[i] == L for i in kl) / len(kl))
            res["vs_" + ref] = r
        res["unlabelled"] = sum(1 for i in ids if J[i] not in LABELS)
        out[nm] = res

    path = HERE / "human_validation_results" / "judges_315.json"
    path.write_text(json.dumps(out, indent=2))
    for nm, r in out.items():
        p = r["vs_pipeline"]
        print(f"{nm:7s} vs pipeline n={p['n']} macroF1={p['macro_f1']:.3f} acc={p['accuracy']:.3f} "
              f"kappa={p['kappa']:.3f} | by source " +
              " ".join(f"{s}={v['accuracy']:.3f}" for s, v in p["by_source"].items()) +
              " | by label " + " ".join(f"{k}={v['accuracy']:.3f}" for k, v in p["by_gold_label"].items()))
        if "vs_A2" in r:
            a = r["vs_A2"]
            print(f"{'':7s} vs A2 n={a['n']} acc={a['accuracy']:.3f} kappa={a['kappa']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
