#!/usr/bin/env python
"""
06_llm_judge_validation.py -- LLM-as-annotator check on the 315-item validation set.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This shows a strong LLM the premise and hypothesis WITHOUT the pipeline's label
and asks it to assign an NLI label, then measures agreement with the automatic
label. It is a machine stand-in for the human validation study of
01_sample_validation_set.py, on exactly the same 315 items.

It does NOT replace that study. An LLM judge and a human annotator can be wrong
in correlated ways -- both may accept a construction shortcut that a careful
reader would reject -- so high agreement here is weaker evidence than high
agreement with people. Report it as an LLM-judge agreement, never as human
validation.

The prompt is ENGLISH over Hebrew premise/hypothesis, deliberately: the paper's
own decoder prompt is Hebrew, so an English instruction gives an independent
reading rather than reproducing the pipeline's framing.

METHOD
------
Per item: premise + hypothesis -> one of entailment / contradiction / neutral.
Temperature 0. Responses cached per (model, item) so reruns cost nothing and
are reproducible. Reports overall agreement, per source, per label, Cohen's
kappa against the automatic labels, and the confusion matrix.

USAGE
    python 06_llm_judge_validation.py --models google/gemini-3.5-flash-lite openai/gpt-5.4-mini
    python 06_llm_judge_validation.py --limit 20        # smoke test
"""

import argparse, json, os, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTER = HERE / "sampled_data" / "validation_set_master.jsonl"
OUT = HERE / "llm_judge_results"
LABELS = ["entailment", "contradiction", "neutral"]

PROMPT = """You are annotating a Natural Language Inference (NLI) dataset. The texts are in Hebrew.

Read the premise and the hypothesis, then decide the relation between them:
- entailment: the hypothesis must be true given the premise
- contradiction: the hypothesis must be false given the premise
- neutral: the hypothesis may be true or false; the premise does not settle it

Assume the premise and hypothesis describe the same situation.

Premise:
{premise}

Hypothesis:
{hypothesis}

Answer with exactly one word: entailment, contradiction, or neutral."""


def client():
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def parse(text):
    t = (text or "").strip().lower()
    for lab in LABELS:                      # exact first, then substring
        if t == lab:
            return lab
    for lab in LABELS:
        if lab in t:
            return lab
    return None


def kappa(a, b):
    """Cohen's kappa between two label sequences."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def run_model(model, rows, cache_path, limit=None, sleep=0.0):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cli = None
    todo = rows[:limit] if limit else rows
    for i, r in enumerate(todo):
        key = r["item_id"]
        if key in cache:
            continue
        if cli is None:
            cli = client()
        prompt = PROMPT.format(premise=r["premise"], hypothesis=r["hypothesis"])
        txt = ""
        for attempt in range(3):
            try:
                resp = cli.chat.completions.create(
                    model=model, temperature=0, max_tokens=8,
                    messages=[{"role": "user", "content": prompt}])
                txt = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    [{key}] failed after 3 attempts: {e}", flush=True)
                else:
                    time.sleep(2 ** attempt)
        cache[key] = txt
        if (i + 1) % 25 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            print(f"    {i+1}/{len(todo)}", flush=True)
        if sleep:
            time.sleep(sleep)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    return cache


def report(model, rows, cache):
    gold, pred, unparsed = [], [], 0
    per_src, per_lab = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    conf = Counter()
    for r in rows:
        raw = cache.get(r["item_id"])
        if raw is None:
            continue
        p = parse(raw)
        if p is None:
            unparsed += 1
            continue
        g = r["label"]
        gold.append(g); pred.append(p)
        conf[(g, p)] += 1
        for d, k in ((per_src, r["dataset"]), (per_lab, g)):
            d[k][1] += 1
            d[k][0] += int(p == g)
    n = len(gold)
    agree = sum(g == p for g, p in zip(gold, pred)) / n if n else 0.0
    res = dict(model=model, n_scored=n, unparsed=unparsed,
               agreement=agree, cohen_kappa=kappa(gold, pred),
               per_source={k: dict(agree=v[0] / v[1], n=v[1]) for k, v in sorted(per_src.items())},
               per_label={k: dict(agree=v[0] / v[1], n=v[1]) for k, v in sorted(per_lab.items())},
               confusion={f"{g}->{p}": c for (g, p), c in sorted(conf.items())},
               pred_distribution=dict(Counter(pred)))
    print(f"\n=== {model} ===")
    print(f"  agreement with pipeline label : {agree:.3f}   (n={n}, unparsed={unparsed})")
    print(f"  Cohen's kappa                 : {res['cohen_kappa']:.3f}")
    print("  per source:", "  ".join(f"{k}={v['agree']:.3f}" for k, v in res["per_source"].items()))
    print("  per label :", "  ".join(f"{k}={v['agree']:.3f}" for k, v in res["per_label"].items()))
    print("  predicted :", res["pred_distribution"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["google/gemini-3.5-flash-lite", "openai/gpt-5.4-mini"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in MASTER.open(encoding="utf-8") if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"validation set: {len(rows)} items"
          f"  sources={len(set(r['dataset'] for r in rows))}"
          f"  labels={dict(Counter(r['label'] for r in rows))}")

    all_res = []
    for m in a.models:
        tag = m.replace("/", "__")
        cache = run_model(m, rows, OUT / f"cache_{tag}.json", a.limit, a.sleep)
        all_res.append(report(m, rows[:a.limit] if a.limit else rows, cache))
    (OUT / "llm_judge_summary.json").write_text(
        json.dumps(all_res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'llm_judge_summary.json'}")


if __name__ == "__main__":
    main()
