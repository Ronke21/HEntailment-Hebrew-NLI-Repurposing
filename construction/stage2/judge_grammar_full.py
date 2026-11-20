#!/usr/bin/env python3
"""
Grammar judge helpers (GPU, vLLM).

edited_sentence() returns the sentence a rule rewrote; the grammar prompt is
applied to that sentence, not to the whole (possibly multi-sentence)
hypothesis. Run directly, it judges grammar alone for the top-k candidates of
each row, with verdicts cached by text. The released corpus uses judge_full.py,
which applies both the grammar and the label check.
Output: data/corpus/<src>/grammar_<tag>[.shardIofN].jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
RESOURCES = REPO / "construction/resources"
OUT = REPO / "data/corpus"
sys.path.insert(0, str(REPO / "construction/stage2"))
sys.path.insert(0, str(REPO / "construction/stage2/utils"))
from judge_candidates import GRAMMAR_PROMPT, rank_candidates   # noqa: E402
import contradiction_utils as cu                                # noqa: E402


def edited_sentence(cand: dict) -> str:
    """The one sentence a rule rewrote; the whole text when it has only one."""
    parts = [x.strip() for x in cu._SENT_SPLIT.split(cand["text"]) if x.strip()]
    i = cand.get("sentence", 0)
    return parts[i] if 0 <= i < len(parts) else cand["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--max-candidates", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=6000)
    ap.add_argument("--shard", default="")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    args = ap.parse_args()
    tag = args.model.split("/")[-1].replace(".", "_")
    d = OUT / args.src
    items = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]   # v2 rules rejected, see 7.9.2
    sfx = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/")); items = items[i::n]; sfx = f".shard{i}of{n}"
    texts = []
    for it in items:
        for c in rank_candidates(it["candidates"], args.max_candidates):
            texts.append(edited_sentence(c))
    texts = sorted(set(texts))
    # text-keyed cache: verdict files from earlier runs, this build's finished/partial files
    cache = {}
    for f0 in list(RESOURCES.glob(f"{args.src}/verdicts_{tag}*.jsonl")) + list(d.glob(f"grammar_{tag}*.jsonl")):
        for l in open(f0, encoding="utf-8"):
            v = json.loads(l)
            if v.get("text") is not None and "gram_ok" in v:
                cache[v["text"]] = {"gram_raw": v.get("gram_raw", ""), "gram_ok": bool(v["gram_ok"])}
    todo = [t for t in texts if t not in cache]
    print(f"[{args.src}{sfx}] {len(items)} rows, {len(texts)} distinct candidate texts, cached {len(texts)-len(todo)}, to judge {len(todo)}", flush=True)
    out_f = d / f"grammar_{tag}{sfx}.jsonl"
    if todo:
        from vllm import LLM, SamplingParams
        import torch
        free_b, total_b = torch.cuda.mem_get_info()
        llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
                  gpu_memory_utilization=max(0.3, min(0.95, free_b * 0.97 / total_b)), max_num_seqs=args.max_num_seqs)
        tok = llm.get_tokenizer()
        def render(t):
            m = [{"role": "user", "content": GRAMMAR_PROMPT.format(text=t)}]
            try: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError: return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        sp = SamplingParams(temperature=0, max_tokens=8); t0 = time.time()
        part = d / f"grammar_{tag}{sfx}.partial.jsonl"
        with open(part, "a", encoding="utf-8") as pf:
            for c0 in range(0, len(todo), args.chunk):
                chunk = todo[c0:c0 + args.chunk]
                outs = llm.generate([render(t) for t in chunk], sp)
                for t, o in zip(chunk, outs):
                    raw = o.outputs[0].text.strip()
                    cache[t] = {"gram_raw": raw, "gram_ok": raw.startswith("תקין")}
                    pf.write(json.dumps({"text": t, **cache[t]}, ensure_ascii=False) + "\n")
                pf.flush(); os.fsync(pf.fileno())
                print(f"[{args.src}{sfx}] {min(c0+len(chunk), len(todo))}/{len(todo)} judged ({time.time()-t0:.0f}s)", flush=True)
    with open(out_f, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t, **cache[t]}, ensure_ascii=False) + "\n")
    (d / f"grammar_{tag}{sfx}.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
