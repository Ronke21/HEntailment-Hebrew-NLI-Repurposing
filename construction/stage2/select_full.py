#!/usr/bin/env python3
"""
Stage 2e: write the corpus (CPU).

Each split of a source is its stage-1 entailment and neutral rows (text and ids
unchanged) plus one judged contradiction per entailment row where one exists:
  * eligible candidate: passes both judge checks (grammar of the edited
    sentence, and label == contradiction for the pair);
  * rules in <src>/exclude_rules.txt are excluded (they leak the label);
  * grounded candidates are preferred; an ungrounded one ships only when no
    grounded candidate of the row is eligible, and every row records which;
  * ranking: edits of two or more tokens first, fact rules before marker
    rules, fewer inserted tokens; no rule may exceed MAX_SHARE of a source's
    contradictions (greedy over a seeded row order);
  * a contradiction row carries source_entailment, rules and n_edit_tokens.
Refuses to write if any row's top-k candidates lack verdicts.
Output: data/corpus/<src>/{train,dev,test}.jsonl, selection_summary.json
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
STAGE1 = REPO / "data/stage1"
OUT = REPO / "data/corpus"
sys.path.insert(0, str(REPO / "construction/stage2"))
from judge_candidates import eligible, load_excluded, rank_candidates   # noqa: E402
from select_and_write import is_marker, rule_key             # noqa: E402
from judge_grammar_full import edited_sentence               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--judge-tag", default="gemma-4-31B-it")
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--max-share", type=float, default=0.25)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    d = OUT / args.src
    items = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]
    # validity verdicts: (item_id, candidate text) -> both checks
    V = {}
    for f in sorted(d.glob(f"validity_{args.judge_tag}*.jsonl")):
        for l in open(f, encoding="utf-8"):
            v = json.loads(l); V[(v["item_id"], v["cand_text"])] = v
    excluded = load_excluded(d)
    print("validity verdicts:", len(V), "| excluded rules:", sorted(excluded))

    missing = 0
    for it in items:
        for c in rank_candidates(eligible(it["candidates"], excluded), args.max_candidates):
            if (it["item_id"], c["text"]) not in V: missing += 1
    if missing and not args.allow_partial:
        print(f"INCOMPLETE: {missing} top-k candidates have no validity verdict; rerun stage B."); raise SystemExit(2)

    rng = random.Random(42); order = list(range(len(items))); rng.shuffle(order)
    counts, chosen, stats = Counter(), {}, Counter()
    n_target = sum(1 for it in items if it["candidates"])
    for k in order:
        it = items[k]
        if not it["candidates"]: stats["no_candidate"] += 1; continue
        pool = eligible(it["candidates"], excluded)
        stats["cand_excluded_rule"] += len(it["candidates"]) - len(pool)
        if not pool: stats["row_all_candidates_excluded"] += 1; continue
        elig = []
        for c in rank_candidates(pool, args.max_candidates):
            v = V.get((it["item_id"], c["text"]))
            if v is None: stats["cand_unjudged"] += 1; continue
            if not v["gram_ok"]: stats["cand_ungrammatical"] += 1; continue
            if not v["label_ok"]: stats["cand_not_contradiction_" + (v["label"] or "unparsed")] += 1; continue
            elig.append(c)
        if not elig: stats["no_grammatical_candidate"] += 1; continue
        elig.sort(key=lambda c: (not c.get("grounded", True), c["n_edit_tokens"] < 2,
                                 is_marker(c), len(c["inserted_tokens"]), rng.random()))
        cap = args.max_share * max(1, n_target)
        under = [c for c in elig if counts[rule_key(c)] < cap]
        pick = under[0] if under else elig[0]
        if not under: stats["over_cap_fallback"] += 1
        counts[rule_key(pick)] += 1; chosen[it["item_id"]] = pick; stats["kept"] += 1

    by_split = defaultdict(list)
    for it in items:
        c = chosen.get(it["item_id"])
        if c: by_split[it["split"]].append({"premise": it["premise"], "hypothesis": c["text"], "label": "contradiction",
                                             "id": f"{it['v2b_id']}_con", "source_entailment": it["entailment"],
                                             "rules": c["rules"], "n_edit_tokens": c["n_edit_tokens"],
                                             "grounded": bool(c.get("grounded", True))})
    counts_out = {}
    for sp in ("train", "dev", "test"):
        rows = []
        for l in open(STAGE1 / args.src / f"{sp}.jsonl", encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r["label"] in ("entailment", "neutral"):
                rows.append({"premise": r["premise"], "hypothesis": r["hypothesis"], "label": r["label"], "id": r.get("id")})
        rows += by_split.get(sp, [])
        random.Random(42).shuffle(rows)
        with open(d / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts_out[sp] = dict(Counter(r["label"] for r in rows)); counts_out[sp]["rows"] = len(rows)
    kept = list(chosen.values())
    summary = {"src": args.src, "judge": args.judge_tag + " (grammar + contradiction label)", "entailment_rows": len(items),
               "contradictions_written": len(kept), "coverage": round(len(kept) / max(1, len(items)), 3),
               "splits": counts_out, "stats": dict(stats),
               "rule_share": {k: round(v / max(1, len(kept)), 3) for k, v in counts.most_common()},
               "edit_tokens": dict(Counter(min(c["n_edit_tokens"], 4) for c in kept)),
               "share_marker_rules": round(sum(is_marker(c) for c in kept) / max(1, len(kept)), 3),
               "share_ungrounded": round(sum(not c.get("grounded", True) for c in kept) / max(1, len(kept)), 3)}
    json.dump(summary, open(d / "selection_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)); print("DONE")


if __name__ == "__main__":
    main()
