#!/usr/bin/env python3
"""
Stage 2c (merge): fold the ungrounded shards into candidates.jsonl. Existing
candidates are tagged grounded=True and keep their text; added ones carry
grounded=False. Backup: candidates.pre_ungrounded.jsonl.
"""
from __future__ import annotations
import argparse, json, shutil
from collections import Counter
from pathlib import Path

OUT = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "data/corpus"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True)
    a = ap.parse_args(); d = OUT / a.src
    shards = sorted(d.glob("ungrounded.shard*.jsonl")) or sorted(d.glob("ungrounded.jsonl"))
    if not shards:
        raise SystemExit("no ungrounded shards")
    add = {}
    for f in shards:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); add.setdefault(r["item_id"], []).extend(r["candidates"])
    bak = d / "candidates.pre_ungrounded.jsonl"
    if not bak.exists():
        shutil.copy2(d / "candidates.jsonl", bak)
    stats = Counter(); tmp = d / "candidates.jsonl.tmp"
    with open(bak, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as g:
        for l in src:
            r = json.loads(l)
            for c in r["candidates"]:
                c["grounded"] = True
            extra = add.get(r["item_id"], [])
            if extra:
                stats["rows_extended"] += 1
                stats["rows_rescued"] += not r["candidates"]
            r["candidates"] = r["candidates"] + extra
            stats["grounded"] += sum(1 for c in r["candidates"] if c["grounded"])
            stats["ungrounded"] += len(extra)
            stats["rows_with_any"] += bool(r["candidates"])
            stats["rows"] += 1
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(d / "candidates.jsonl")
    json.dump({"src": a.src, **stats}, open(d / "ungrounded_merge_summary.json", "w", encoding="utf-8"), indent=1)
    (d / "C.DONE").write_text(json.dumps({"src": a.src, **stats}) + "\n")
    print(json.dumps({"src": a.src, **stats}, indent=1)); print("DONE")


if __name__ == "__main__":
    main()
