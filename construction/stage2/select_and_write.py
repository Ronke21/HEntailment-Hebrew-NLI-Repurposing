#!/usr/bin/env python3
"""
Selection helpers shared with select_full.py: rule keys and the fact-rule /
marker-rule split used for ranking (marker rules insert label-bearing words such
as negation, quantity or modality markers; fact rules change a number, entity,
time, degree or lexical item).
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = REPO / "construction/resources"
MARKER_FAMILIES = {"negation", "quantity", "modality"}
FAMILY = {
    "number_substitution": "number", "number_substitution_heb": "number",
    "verb_negation_v3": "negation", "existential_negation": "negation",
    "quantifier_flip": "quantity", "modal_flip": "modality", "epistemic_certainty": "modality",
    "temporal_inversion": "time", "superlative_flip": "degree", "scope_severity": "degree", "pace_velocity": "degree",
    "adjective_antonym": "lexical", "causal_reversal": "lexical", "geo_directional_flip": "lexical",
    "decision_inversion": "outcome", "financial_outcome": "outcome", "legal_status": "outcome",
    "inclusion_exclusion": "outcome", "factional_alliance": "outcome", "state_transition": "outcome",
}
def family(n): return "entity" if n.startswith("entity_swap") else FAMILY.get(n, n)
def is_marker(c): return any(family(r) in MARKER_FAMILIES for r in c["rules"])
def rule_key(c): return "+".join(c["rules"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="heq")
    ap.add_argument("--judge-tag", default="gemma-4-31B-it")
    ap.add_argument("--max-share", type=float, default=0.25)
    ap.add_argument("--keep-unjudged", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="write the corpus even if some items were never judged")
    ap.add_argument("--exclude-rules", default="", help="comma list of rule names the gate rejected")
    args = ap.parse_args()
    d = OUT / args.src
    items = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]
    V = defaultdict(dict)
    vfiles = sorted(d.glob(f"verdicts_{args.judge_tag}*.jsonl"))      # main file and/or shards
    for vf in vfiles:
        for l in open(vf, encoding="utf-8"):
            v = json.loads(l); V[v["item_id"]][v["kind"]] = v
    print("verdict files:", [f.name for f in vfiles])
    excluded = {r for r in args.exclude_rules.split(",") if r}
    # sbatch --export splits its value on commas and truncates the list; the
    # authoritative exclusion list therefore lives in <src>/exclude_rules.txt
    xf = d / "exclude_rules.txt"
    if xf.exists():
        excluded |= {r.strip() for r in xf.read_text().split() if r.strip()}
    print("excluded rules:", sorted(excluded))

    # completeness guard: a judge shard killed at the 4h wall leaves some items
    # unjudged, and writing the corpus then would silently drop them. Refuse.
    need = [it for it in items if it["candidates"]]
    missing = [it["item_id"] for it in need if it["item_id"] not in V]
    if missing and not args.allow_partial:
        print(f"INCOMPLETE: {len(missing)} of {len(need)} items with candidates have no verdicts "
              f"(e.g. {missing[:3]}). Rerun the judge (it resumes from the partial file) or pass "
              f"--allow-partial deliberately."); raise SystemExit(2)

    rng = random.Random(42); order = list(range(len(items))); rng.shuffle(order)
    counts, chosen, stats = Counter(), {}, Counter()
    total_target = sum(1 for it in items if it["candidates"])
    for k in order:
        it = items[k]; v = V.get(it["item_id"], {})
        if not args.keep_unjudged:
            if not (v.get("entailment", {}).get("label_ok") and v.get("entailment", {}).get("gram_ok")):
                stats["dropped_entailment_failed_judge"] += 1; continue
            if it.get("neutral") and not (v.get("neutral", {}).get("label_ok") and v.get("neutral", {}).get("gram_ok")):
                stats["dropped_neutral_failed_judge"] += 1; continue
        elig = []
        for ci, c in enumerate(it["candidates"]):
            vv = v.get(f"cand{ci}")
            if not vv: stats["candidate_unjudged"] += 1; continue
            if any(r in excluded for r in c["rules"]): stats["candidate_rule_excluded"] += 1; continue
            if vv["label_ok"] and vv["gram_ok"]: elig.append(c)
            else: stats["candidate_rejected_by_judge"] += 1
        if not elig:
            stats["dropped_no_valid_candidate" if it["candidates"] else "dropped_no_candidate"] += 1; continue
        # rank: >=2 edits first, then fact over marker, then fewer inserted tokens, then seeded tie-break
        elig.sort(key=lambda c: (c["n_edit_tokens"] < 2, is_marker(c), len(c["inserted_tokens"]), rng.random()))
        cap = args.max_share * max(1, total_target)
        under = [c for c in elig if counts[rule_key(c)] < cap]
        pick = under[0] if under else elig[0]
        if not under: stats["over_cap_fallback"] += 1
        counts[rule_key(pick)] += 1; chosen[it["item_id"]] = pick; stats["kept"] += 1

    results, rows = [], defaultdict(list)
    for it in items:
        c = chosen.get(it["item_id"])
        rec = {**it, "chosen": c, "kept": c is not None}
        results.append(rec)
        if c is None: continue
        rows[it["split"]].append({"premise": it["premise"], "hypothesis": it["entailment"], "label": "entailment"})
        rows[it["split"]].append({"premise": it["premise"], "hypothesis": c["text"], "label": "contradiction",
                                  "rules": c["rules"], "n_edit_tokens": c["n_edit_tokens"]})
        if it.get("neutral"):
            rows[it["split"]].append({"premise": it["premise"], "hypothesis": it["neutral"], "label": "neutral"})
    for sp in ("train", "dev", "test"):
        with open(d / f"{sp}.jsonl", "w", encoding="utf-8") as f:
            for r in rows.get(sp, []): f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(results, open(d / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    kept = [c for c in chosen.values()]
    summary = {"src": args.src, "judge": args.judge_tag, "items": len(items), "kept": len(kept),
               "rows": {sp: len(rows.get(sp, [])) for sp in ("train", "dev", "test")},
               "stats": dict(stats), "rule_share": {k: round(v / max(1, len(kept)), 3) for k, v in counts.most_common()},
               "edit_tokens": dict(Counter(min(c["n_edit_tokens"], 4) for c in kept)),
               "share_marker_rules": round(sum(is_marker(c) for c in kept) / max(1, len(kept)), 3)}
    json.dump(summary, open(d / "selection_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)); print("DONE")


if __name__ == "__main__":
    main()
