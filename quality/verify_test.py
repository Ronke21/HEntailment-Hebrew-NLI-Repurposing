#!/usr/bin/env python3
"""
verify_test.py -- independent-judge pass over the whole HEntailment test split
(release `verified` subset).

Qwen3.6-27B, a family never used to build the corpus, labels every test pair
with the same prompt, rendering and parser as second_judge_grounding.py (the
2,000-contradiction check reported in the paper). An item is `verified` when
the judge's label equals the corpus label; an unparsable answer counts as not
verified.

Resumable: rows are appended per chunk to rows.jsonl and already-judged ids are
skipped on restart.

Output: corpus_build_full/_verified_test/{rows.jsonl, report.json}
"""
from __future__ import annotations
import argparse, collections, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = Path(os.environ.get("HENT_CORPUS", HERE.parents[1] / "data/corpus"))
DEST = CORPUS / "_verified_test"
SOURCES = ("heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase")
LABELS = ("entailment", "contradiction", "neutral")
sys.path.insert(0, str(HERE.parents[1] / "construction/stage2"))


def load_units():
    units = []
    for src in SOURCES:
        for l in open(CORPUS / src / "test.jsonl", encoding="utf-8"):
            r = json.loads(l)
            u = {"id": r["id"], "src": src, "label": r["label"],
                 "premise": r["premise"], "hypothesis": r["hypothesis"]}
            for k in ("rules", "rule", "grounded"):
                if k in r:
                    u[k] = r[k]
            units.append(u)
    return units


def report(rows):
    res = collections.defaultdict(collections.Counter)
    for r in rows:
        ok = r["judge_label"] == r["label"]
        for k in ("ALL", f"ALL|{r['label']}", r["src"], f"{r['src']}|{r['label']}"):
            res[k]["n"] += 1
            res[k]["verified"] += ok
            res[k]["unparsed"] += r["judge_label"] is None
    return {k: {**dict(v), "rate": round(v["verified"] / v["n"], 4)} for k, v in sorted(res.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=1000)
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    from judge_candidates import LABEL_PROMPT, MAX_PREMISE_WORDS, parse_label

    units = load_units()
    rows_f = DEST / "rows.jsonl"
    done = {}
    if rows_f.exists():
        for l in open(rows_f, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    todo = [u for u in units if u["id"] not in done]
    print(f"{len(units)} test items, {len(done)} already judged, {len(todo)} to go", flush=True)

    if todo:
        from vllm import LLM, SamplingParams
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
        frac = max(0.30, min(0.90, free_b * 0.90 / total_b))
        llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
                  gpu_memory_utilization=frac, max_num_seqs=a.max_num_seqs,
                  enable_prefix_caching=True)
        tok = llm.get_tokenizer()

        def render(c):
            m = [{"role": "user", "content": c}]
            try:
                return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

        sp = SamplingParams(temperature=0, max_tokens=12)
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            outs = llm.generate([render(LABEL_PROMPT.format(
                premise=" ".join(u["premise"].split()[:MAX_PREMISE_WORDS]), hypothesis=u["hypothesis"]))
                for u in chunk], sp)
            with open(rows_f, "a", encoding="utf-8") as f:
                for u, o in zip(chunk, outs):
                    r = {k: v for k, v in u.items() if k not in ("premise", "hypothesis")}
                    r["judge_raw"] = o.outputs[0].text
                    r["judge_label"] = parse_label(o.outputs[0].text)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done[r["id"]] = r
            print(f"judged {min(s + a.chunk, len(todo))}/{len(todo)}", flush=True)

    rows = [done[u["id"]] for u in units if u["id"] in done]
    rep = report(rows)
    json.dump({"model": a.model, "n_items": len(units), "n_judged": len(rows), "report": rep},
              open(DEST / "report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ["ALL"] + [f"ALL|{l}" for l in LABELS] + list(SOURCES):
        if k in rep:
            print(k, rep[k], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
