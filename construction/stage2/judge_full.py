#!/usr/bin/env python3
"""
Stage 2d: the construction judge on contradiction candidates (GPU, vLLM,
gemma-4-31B-it).

A candidate is eligible only if the judge says both
  * the edited sentence is well-formed Hebrew (GRAMMAR_PROMPT), and
  * the (premise, hypothesis) pair is a contradiction (LABEL_PROMPT).
Entailments and neutrals are not judged. The top-K ranked candidates of each
row are judged (--max-candidates, 8 by default), after rules in
<src>/exclude_rules.txt are dropped. Chunked, resumable and sharded over rows.

Output: data/corpus/<src>/validity_<tag>[.shardIofN].jsonl
        {item_id, cand_text, edited_sentence, gram_ok, label, label_ok}
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = REPO / "data/corpus"
RESOURCES = REPO / "construction/resources"
sys.path.insert(0, str(REPO / "construction/stage2"))
sys.path.insert(0, str(REPO / "construction/stage2/utils"))
from judge_candidates import (GRAMMAR_PROMPT, LABEL_PROMPT, MAX_PREMISE_WORDS,  # noqa: E402
                              eligible, load_excluded, parse_label, rank_candidates)
from judge_grammar_full import edited_sentence                                   # noqa: E402


def load_rows(d: Path, shard: str):
    rows = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]
    sfx = ""
    if shard:
        i, n = (int(x) for x in shard.split("/")); rows = rows[i::n]; sfx = f".shard{i}of{n}"
    return [r for r in rows if r["candidates"]], sfx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=3000, help="candidates per judged+written chunk")
    ap.add_argument("--shard", default="")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    args = ap.parse_args()
    tag = args.model.split("/")[-1].replace(".", "_")
    d = OUT / args.src
    rows, sfx = load_rows(d, args.shard)

    excluded = load_excluded(d)
    units = []          # (item_id, premise, cand_text, edited_sentence)
    n_excl = n_dead = 0
    for r in rows:
        prem = " ".join(r["premise"].split()[:MAX_PREMISE_WORDS])
        elig = eligible(r["candidates"], excluded)
        n_excl += len(r["candidates"]) - len(elig)
        if not elig:
            n_dead += 1; continue
        for c in rank_candidates(elig, args.max_candidates):
            units.append((r["item_id"], prem, c["text"], edited_sentence(c)))

    out_f = d / f"validity_{tag}{sfx}.jsonl"
    part_f = d / f"validity_{tag}{sfx}.partial.jsonl"
    done = set()
    for f0 in (out_f, part_f):
        if f0.exists():
            for l in open(f0, encoding="utf-8"):
                v = json.loads(l); done.add((v["item_id"], v["cand_text"]))
    # grammar verdicts are text-keyed and premise-independent: reuse every earlier pass
    gram_cache = {}
    for f0 in list(d.glob(f"grammar_{tag}*.jsonl")) + list(RESOURCES.glob(f"{args.src}/verdicts_{tag}*.jsonl")):
        for l in open(f0, encoding="utf-8"):
            v = json.loads(l)
            if v.get("text") is not None and "gram_ok" in v:
                gram_cache[v["text"]] = bool(v["gram_ok"])
    todo = [u for u in units if (u[0], u[2]) not in done]
    print(f"[{args.src}{sfx}] {len(rows)} rows, excluded rules dropped {n_excl} candidates and "
          f"{n_dead} rows outright; {len(units)} candidates to judge (top-{args.max_candidates}), "
          f"already judged {len(units)-len(todo)}, grammar cache {len(gram_cache)}", flush=True)
    if not todo:
        os.replace(part_f, out_f) if part_f.exists() and not out_f.exists() else None
        (d / f"validity_{tag}{sfx}.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n")); print("DONE", flush=True); return

    from vllm import LLM, SamplingParams
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    frac = max(0.30, min(0.85, free_b * 0.90 / total_b))
    print(f"free={free_b/2**30:.1f}GiB -> gpu_memory_utilization={frac:.3f}", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=frac, max_num_seqs=args.max_num_seqs)
    tok = llm.get_tokenizer()

    def render(content):
        m = [{"role": "user", "content": content}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    sp = SamplingParams(temperature=0, max_tokens=12)
    t0 = time.time(); n_ok = 0
    with open(part_f, "a", encoding="utf-8") as pf:
        for c0 in range(0, len(todo), args.chunk):
            chunk = todo[c0:c0 + args.chunk]
            lab_out = llm.generate([render(LABEL_PROMPT.format(premise=p, hypothesis=t)) for _, p, t, _ in chunk], sp)
            need_gram = sorted({s for _, _, _, s in chunk if s not in gram_cache})
            if need_gram:
                gr = llm.generate([render(GRAMMAR_PROMPT.format(text=s)) for s in need_gram], sp)
                for s, o in zip(need_gram, gr):
                    gram_cache[s] = o.outputs[0].text.strip().startswith("תקין")
            for (iid, _p, ctext, sent), o in zip(chunk, lab_out):
                lab = parse_label(o.outputs[0].text)
                rec = {"item_id": iid, "cand_text": ctext, "edited_sentence": sent,
                       "gram_ok": bool(gram_cache.get(sent, False)),
                       "label": lab, "label_ok": lab == "contradiction"}
                n_ok += rec["label_ok"] and rec["gram_ok"]
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush(); os.fsync(pf.fileno())
            print(f"[{args.src}{sfx}] {min(c0+len(chunk), len(todo))}/{len(todo)} judged, "
                  f"{n_ok} passed both ({time.time()-t0:.0f}s)", flush=True)
    seen, merged = set(), []
    for f0 in (out_f, part_f):
        if f0.exists():
            for l in open(f0, encoding="utf-8"):
                v = json.loads(l); k = (v["item_id"], v["cand_text"])
                if k not in seen:
                    seen.add(k); merged.append(v)
    with open(out_f, "w", encoding="utf-8") as f:
        for v in merged:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    (d / f"validity_{tag}{sfx}.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    print(f"[{args.src}{sfx}] wrote {len(merged)} verdicts", flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
