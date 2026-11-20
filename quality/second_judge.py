#!/usr/bin/env python3
"""
Independent second judge on shipped contradictions (GPU, vLLM).

Qwen3.6-27B, a model family never used to build the corpus, labels a sample of
contradictions (per source, split by the grounded flag) with the same prompt and
parser as the construction judge. Its agreement rate is the contradiction-validity
figure reported in the paper; the grounded / ungrounded split shows whether the
ungrounded contradictions are as valid as the grounded ones.
Output: data/corpus/_second_judge/{report.json, rows.jsonl}
"""
from __future__ import annotations
import argparse, collections, json, random, sys
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1])
OUT = REPO / "data/corpus"
DEST = OUT / "_second_judge"
sys.path.insert(0, str(REPO / "construction/stage2"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--per-arm-per-src", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    from judge_candidates import LABEL_PROMPT, MAX_PREMISE_WORDS, parse_label
    by = collections.defaultdict(list)
    for src in ("heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"):
        for sp in ("train", "dev", "test"):
            f = OUT / src / f"{sp}.jsonl"
            if not f.exists():
                continue
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                if r["label"] == "contradiction":
                    by[(src, bool(r.get("grounded", True)))].append(r)
    units = []
    for k, v in by.items():
        random.Random(23).shuffle(v)
        for r in v[:a.per_arm_per_src]:
            units.append({"src": k[0], "grounded": k[1], "id": r.get("id"),
                          "premise": r["premise"], "hypothesis": r["hypothesis"],
                          "rules": r.get("rules")})
    print(f"{len(units)} contradictions to re-judge; "
          f"{ {f'{s}/{ (chr(103) if g else chr(117)) }': len(v) for (s, g), v in sorted(by.items()) } }", flush=True)

    from vllm import LLM, SamplingParams
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    frac = max(0.30, min(0.85, free_b * 0.90 / total_b))
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=frac, max_num_seqs=a.max_num_seqs)
    tok = llm.get_tokenizer()

    def render(c):
        m = [{"role": "user", "content": c}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    outs = llm.generate([render(LABEL_PROMPT.format(
        premise=" ".join(u["premise"].split()[:MAX_PREMISE_WORDS]), hypothesis=u["hypothesis"])) for u in units],
        SamplingParams(temperature=0, max_tokens=12))
    res = collections.defaultdict(collections.Counter)
    with open(DEST / "rows.jsonl", "w", encoding="utf-8") as f:
        for u, o in zip(units, outs):
            lab = parse_label(o.outputs[0].text)
            f.write(json.dumps({**u, "second_judge": lab}, ensure_ascii=False) + "\n")
            arm = "grounded" if u["grounded"] else "ungrounded"
            for k in (f"ALL|{arm}", f"{u['src']}|{arm}"):
                res[k]["n"] += 1; res[k][lab or "unparsed"] += 1
    report = {k: {**dict(v), "agrees_contradiction": round(v["contradiction"] / max(1, v["n"]), 3)}
              for k, v in sorted(res.items())}
    json.dump({"model": a.model, "report": report}, open(DEST / "report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(report, indent=1, ensure_ascii=False)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
