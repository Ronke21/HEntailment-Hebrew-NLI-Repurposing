#!/usr/bin/env python3
"""
Stage 2b: candidates for entailment rows whose hypothesis contains a QASem
argument placeholder (מישהו / משהו / מישהי / כלשהו / כלשהי).

Stage 2a skips these rows. They get candidates here and pass through the same
judge, so a placeholder contradiction ships only if the judge calls the pair a
contradiction. The seed is derived from the item id, so a rerun reproduces it.
Rewrites data/corpus/<src>/candidates.jsonl in place (backup kept).
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from collections import defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = REPO / "data/corpus"
RESOURCES = REPO / "construction/resources"
sys.path.insert(0, str(REPO / "construction/stage2"))
import gen_candidates_full as F   # noqa: E402
import gen_candidates as G        # noqa: E402


def seed_of(item_id: str) -> int:
    return int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--no-grounding", action="store_true")
    a = ap.parse_args()
    d = OUT / a.src
    f = d / "candidates.jsonl"
    bak = d / "candidates.pre_placeholder.jsonl"
    if not bak.exists():
        shutil.copy2(f, bak)
    pool = json.load(open(RESOURCES / a.src / "entity_pool.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(bak, encoding="utf-8")]
    todo = [i for i, r in enumerate(rows)
            if not r["candidates"] and any(p in r["entailment"] for p in F.PLACEHOLDERS)]
    print(f"[{a.src}] {len(rows)} rows, {len(todo)} placeholder rows to generate", flush=True)
    t0 = time.time(); n_with = n_ungrounded = 0; by_rule = defaultdict(int)
    for n, i in enumerate(todo):
        r = rows[i]
        rng = G.RecordingRNG(seed_of(r["item_id"]))
        c = F.candidates_for(r["entailment"], r["premise"], pool, rng)
        if not a.no_grounding:
            ps = F._stems(r["premise"]); before = len(c)
            c = [x for x in c if F.grounded(r["entailment"], x, ps)]
            n_ungrounded += before - len(c)
        rows[i]["candidates"] = c
        rows[i]["placeholder_row"] = True
        n_with += bool(c)
        for x in c: by_rule["+".join(x["rules"])] += 1
        if (n + 1) % 1000 == 0:
            print(f"[{a.src}] {n+1}/{len(todo)}, {n_with} with a candidate ({time.time()-t0:.0f}s)", flush=True)
    tmp = f.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(f)
    summary = {"src": a.src, "placeholder_rows": len(todo), "rows_with_candidate": n_with,
               "candidates_dropped_ungrounded": n_ungrounded,
               "candidates_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
               "elapsed_s": round(time.time() - t0)}
    json.dump(summary, open(d / "placeholder_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates_by_rule"}, ensure_ascii=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
