#!/usr/bin/env python3
"""
Stage 2c: add ungrounded candidates (rule edits whose replaced token does not
occur in the premise), tagged grounded=False.

Grounding is a preference, not a filter: selection puts grounded candidates
first, and an ungrounded one ships only when no grounded candidate of the row
passes the judge. Candidates already on disk keep their exact text.
Output: data/corpus/<src>/ungrounded[.shardIofN].jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from collections import defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = REPO / "data/corpus"
RESOURCES = REPO / "construction/resources"
sys.path.insert(0, str(REPO / "construction/stage2"))
import gen_candidates_full as F   # noqa: E402
import gen_candidates as G        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--shard", default="", help="I/N over rows")
    a = ap.parse_args()
    d = OUT / a.src
    pool = json.load(open(RESOURCES / a.src / "entity_pool.json", encoding="utf-8"))
    rows = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]
    sfx = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/")); rows = rows[i::n]; sfx = f".shard{i}of{n}"
    t0 = time.time(); n_add = 0; n_rows_add = 0; n_rows_rescued = 0; by_rule = defaultdict(int)
    print(f"[{a.src}{sfx}] {len(rows)} rows", flush=True)
    with open(d / f"ungrounded{sfx}.jsonl", "w", encoding="utf-8") as g:
        for k, r in enumerate(rows):
            have = {c["text"] for c in r["candidates"]}
            rng = G.RecordingRNG(int(hashlib.md5(("ug/" + r["item_id"]).encode("utf-8")).hexdigest()[:8], 16))
            ps = F._stems(r["premise"])
            new = []
            for c in F.candidates_for(r["entailment"], r["premise"], pool, rng):
                if c["text"] in have or F.grounded(r["entailment"], c, ps):
                    continue
                have.add(c["text"]); c["grounded"] = False; new.append(c)
                by_rule["+".join(c["rules"])] += 1
            if new:
                n_add += len(new); n_rows_add += 1
                n_rows_rescued += not r["candidates"]
                g.write(json.dumps({"item_id": r["item_id"], "candidates": new}, ensure_ascii=False) + "\n")
            if (k + 1) % 2000 == 0:
                print(f"[{a.src}{sfx}] {k+1}/{len(rows)} rows, +{n_add} candidates ({time.time()-t0:.0f}s)", flush=True)
    summary = {"src": a.src, "shard": a.shard, "rows": len(rows), "ungrounded_added": n_add,
               "rows_gaining_candidates": n_rows_add, "rows_with_nothing_before": n_rows_rescued,
               "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])), "elapsed_s": round(time.time() - t0)}
    json.dump(summary, open(d / f"ungrounded_summary{sfx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_rule"}, ensure_ascii=False)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
