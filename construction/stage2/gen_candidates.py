#!/usr/bin/env python3
"""
Rule machinery for contradiction candidates, and the per-source entity pool.

Library used by gen_candidates_full.py: the rule set (utils/contradiction_utils
and the dictabert-joint tagger in utils/pos_tagger), compound two-rule edits,
and per-candidate bookkeeping (rule names, edited tokens, inserted tokens).
Run directly with --pool-only to build construction/resources/<src>/entity_pool.json
(named entities that occur at least three times in the source's premises; the
entity-swap rule draws replacements from it).
"""
from __future__ import annotations
import argparse, difflib, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
STAGE1 = REPO / "data/stage1"
OUT = REPO / "construction/resources"
sys.path.insert(0, str(REPO / "construction/stage2/utils"))
import contradiction_utils as cu          # noqa: E402
import pos_tagger                          # noqa: E402

SEED = 42
MAX_COMPOUND_PER_ITEM = 3

# rule families: a compound must combine two different families
FAMILY = {
    "number_substitution": "number", "number_substitution_heb": "number", "number_plausible": "number",
    "verb_negation_v3": "negation", "existential_negation": "negation",
    "quantifier_flip": "quantity", "modal_flip": "modality", "epistemic_certainty": "modality",
    "temporal_inversion": "time", "superlative_flip": "degree", "scope_severity": "degree", "pace_velocity": "degree",
    "adjective_antonym": "lexical", "causal_reversal": "lexical", "geo_directional_flip": "lexical",
    "decision_inversion": "outcome", "financial_outcome": "outcome", "legal_status": "outcome",
    "inclusion_exclusion": "outcome", "factional_alliance": "outcome", "state_transition": "outcome",
}
def family(name: str) -> str:
    if name.startswith("entity_swap"): return "entity"
    return FAMILY.get(name, name)


class RecordingRNG(random.Random):
    """apply_heuristics() collects every applicable (candidate, rule) pair and
    then calls rng.choice(list) exactly once on that list. Record the list at
    that call and return the usual random pick; other choice()/shuffle()
    calls (entity-pool draws inside the rules) behave normally."""
    def __init__(self, seed):
        super().__init__(seed); self.last = None
    def choice(self, seq):
        if seq and isinstance(seq[0], tuple) and len(seq[0]) == 2 and isinstance(seq[0][1], str):
            self.last = list(seq)
        return super().choice(seq)


_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(%?)(?![\w.,])")
OLD_NUMBER_RULES = {"number_substitution", "number_substitution_heb"}


def plausible_number(sentence: str, rng: random.Random):
    """Replacement for contradiction_utils' number rule, which multiplied
    the number by 10 (59 -> 590 euro, '6 June' -> '60 June', 10% -> 100%);
    the per-rule gate read those implausible round values at 0.81. Here the
    replacement keeps magnitude and format: years 1500-2030 shift by 1-9,
    percentages stay in 1-100, other integers get a different value with the
    same digit count (thousands separators preserved)."""
    ms = list(_NUM.finditer(sentence))
    if not ms:
        return None
    m = rng.choice(ms)
    raw, pct = m.group(1), m.group(2); val = int(raw.replace(",", ""))
    if pct:
        new = rng.choice([v for v in range(1, 101) if v != val and abs(v - val) >= max(2, val // 4)] or [val + 1])
    elif 1500 <= val <= 2030 and not raw.startswith("0"):
        new = val + rng.choice([d for d in range(-9, 10) if d != 0])
    else:
        lo, hi = 10 ** (len(raw.replace(",", "")) - 1), 10 ** len(raw.replace(",", "")) - 1
        if lo == 1: lo = 0
        cands = [v for v in range(lo, hi + 1) if v != val and abs(v - val) >= max(1, val // 5)] if hi - lo < 5000 else None
        new = rng.choice(cands) if cands else (val + rng.choice([-1, 1]) * max(1, val // 3))
    out = f"{new:,}" if "," in raw else str(new)
    return sentence[:m.start()] + out + pct + sentence[m.end():], "number_plausible"


def all_candidates(sentence: str, premise: str, pool: dict, rng: RecordingRNG, allowed=None):
    rng.last = None
    cu.apply_heuristics(sentence, rng=rng, premise=premise, allowed=allowed, entity_pool=pool)
    cands = [(t, n) for t, n in (rng.last or []) if n not in OLD_NUMBER_RULES]
    if allowed is None or "number_plausible" in allowed:
        r = plausible_number(sentence, rng)
        if r: cands.append(r)
    return cands


def edit_info(orig: str, new: str) -> dict:
    a, b = orig.split(), new.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    changed_a, changed_b, inserted = set(), set(), []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        changed_a.update(range(i1, i2)); changed_b.update(range(j1, j2)); inserted += b[j1:j2]
    return {"n_edit_tokens": max(len(changed_a), len(changed_b)), "changed_orig_idx": sorted(changed_a),
            "inserted_tokens": inserted}


def load_items(src: str, train_cap: int | None):
    """Item selection: one row per
    (premise, label); train needs entailment+neutral (capped, seeded shuffle),
    dev/test need entailment."""
    def groups(split):
        by = defaultdict(dict)
        for l in open(STAGE1 / src / f"{split}.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l); by[r["premise"]][r["label"]] = r
        return by
    rng = random.Random(SEED); out = {}
    tr = [v for v in groups("train").values() if "entailment" in v and "neutral" in v]
    tr.sort(key=lambda v: v["entailment"]["premise"]); rng.shuffle(tr)
    out["train"] = tr[:train_cap] if train_cap else tr
    for sp in ("dev", "test"):
        out[sp] = [v for v in groups(sp).values() if "entailment" in v]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="heq")
    ap.add_argument("--train-cap", type=int, default=0, help="0 = all eligible train premises")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool-only", action="store_true", help="build entity_pool.json and exit")
    ap.add_argument("--shard", default="", help="I/N: process items i%N==I, write candidates.shardIofN.jsonl (cat shards in order afterwards)")
    args = ap.parse_args()
    out_dir = OUT / args.src; out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = load_items(args.src, args.train_cap or None)
    if args.limit:
        items = {k: v[:args.limit] for k, v in items.items()}
    n_items = sum(len(v) for v in items.values())
    print(f"[{args.src}] items: { {k: len(v) for k, v in items.items()} }", flush=True)

    premises = sorted({g["entailment"]["premise"] for v in items.values() for g in v})
    pool_f = out_dir / "entity_pool.json"
    if pool_f.exists():
        pool = json.load(open(pool_f, encoding="utf-8"))
    else:
        print(f"[{args.src}] entity pool from {len(premises)} premises (dictabert-joint, CPU)...", flush=True)
        pool = pos_tagger.build_entity_pool(premises, min_count=3)
        ok = re.compile(r"^[\u05d0-\u05ea][\u05d0-\u05ea\"'\u05f3\u05f4\s\-]*[\u05d0-\u05ea\"'\u05f3\u05f4]$")
        pool = {k: [e for e in v if ok.match(e) and len(e) >= 2] for k, v in pool.items()}
        pool = {k: v for k, v in pool.items() if len(v) >= 3}
        json.dump(pool, open(pool_f, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[{args.src}] pool: { {k: len(v) for k, v in pool.items()} }  ({time.time()-t0:.0f}s)", flush=True)
    if args.pool_only:
        print("DONE (pool only)", flush=True); return
    shard_i, shard_n, sfx = 0, 1, ""
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/")); sfx = f".shard{shard_i}of{shard_n}"

    rule_counts, n_with, n_comp = defaultdict(int), 0, 0
    with open(out_dir / f"candidates{sfx}.jsonl", "w", encoding="utf-8") as f:
        idx = 0
        for split, groups_ in items.items():
            for g in groups_:
                if idx % shard_n != shard_i:
                    idx += 1; continue
                ent = g["entailment"]; premise = ent["premise"]; hyp = ent["hypothesis"]
                rng = RecordingRNG(SEED + idx)
                sents = [s.strip() for s in cu._SENT_SPLIT.split(hyp) if s.strip()]
                cands = []
                for si, sent in enumerate(sents):
                    if len(sent.split()) < 4 and len(sents) > 1:
                        continue
                    for text1, name1 in all_candidates(sent, premise, pool, rng):
                        full = list(sents); full[si] = text1; full1 = " ".join(full)
                        e1 = edit_info(hyp, full1)
                        cands.append({"text": full1, "rules": [name1], "sentence": si, **e1})
                # compound: second rule, different family, disjoint span, on the same sentence
                singles = [c for c in cands]
                for c in singles:
                    if sum(1 for x in cands if len(x["rules"]) > 1) >= MAX_COMPOUND_PER_ITEM:
                        break
                    si = c["sentence"]; sents1 = [s.strip() for s in cu._SENT_SPLIT.split(c["text"]) if s.strip()]
                    if si >= len(sents1): continue
                    fam1 = family(c["rules"][0])
                    allowed = {n for n in FAMILY if FAMILY[n] != fam1} | ({"entity_swap_v3_per", "entity_swap_v3_gpe", "entity_swap_v3_org", "entity_swap_v3_timex"} if fam1 != "entity" else set())
                    for text2, name2 in all_candidates(sents1[si], premise, pool, rng, allowed=allowed):
                        if family(name2) == fam1: continue
                        full = list(sents1); full[si] = text2; full2 = " ".join(full)
                        e2 = edit_info(hyp, full2)
                        e_step = edit_info(c["text"], full2)
                        # spans must be disjoint: the second edit must not touch the first edit's tokens
                        if set(e_step["changed_orig_idx"]) & set(edit_info(hyp, c["text"])["changed_orig_idx"]):
                            continue
                        if e2["n_edit_tokens"] < 2 or full2 == c["text"]:
                            continue
                        cands.append({"text": full2, "rules": c["rules"] + [name2], "sentence": si, **e2})
                        break
                # dedupe by text
                seen, uniq = set(), []
                for c in cands:
                    if c["text"] not in seen and c["text"] != hyp:
                        seen.add(c["text"]); uniq.append(c)
                for c in uniq:
                    rule_counts["+".join(c["rules"])] += 1
                n_with += bool(uniq); n_comp += sum(1 for c in uniq if len(c["rules"]) > 1)
                rec = {"item_id": f"{args.src}_{split}_{idx:05d}", "src": args.src, "split": split,
                       "premise": premise, "entailment": hyp, "candidates": uniq}
                if "neutral" in g:
                    rec["neutral"] = g["neutral"]["hypothesis"]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); idx += 1
                if idx % 500 == 0:
                    print(f"[{args.src}] {idx}/{n_items} items, {n_with} with >=1 candidate ({time.time()-t0:.0f}s)", flush=True)
    summary = {"src": args.src, "items": n_items, "items_with_candidate": n_with, "compound_candidates": n_comp,
               "candidates_by_rule": dict(sorted(rule_counts.items(), key=lambda kv: -kv[1])), "elapsed_s": round(time.time() - t0)}
    json.dump(summary, open(out_dir / f"candidates_summary{sfx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1)); print("DONE", flush=True)


if __name__ == "__main__":
    main()
