#!/usr/bin/env python3
"""
Stage 2a: rule-based contradiction candidates for every entailment hypothesis
of a source (CPU, sharded).

Entailments and neutrals come from stage 1 unchanged; only contradictions are
generated here. All applicable rules are applied, sentence by sentence, with the
edited sentence spliced back into the hypothesis so length is not a cue. A
second rule from another family may edit a different span (compound edit), so
a contradiction can differ from its entailment by two or more tokens. This
stage keeps grounded candidates only (a token the rule replaced appears in the
premise); stage 2c adds the ungrounded ones. Excluded rules are dropped later,
before judging (judge_full.py) and at selection (select_full.py).

Output: data/corpus/<src>/candidates[.shardIofN].jsonl, one line per entailment
row; item id = "<src>/<split>/<stage-1 id>".
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
STAGE1 = REPO / "data/stage1"
RESOURCES = REPO / "construction/resources"          # entity pools live here
OUT = REPO / "data/corpus"
sys.path.insert(0, str(REPO / "construction/stage2"))
import gen_candidates as G                              # noqa: E402  (rules, RecordingRNG, edit_info, family)
import contradiction_utils as cu                        # noqa: E402  (imported by G via the utils path)

MAX_COMPOUND_PER_ITEM = 3
HEB = __import__("re").compile(r"[\u0590-\u05ff]")
PLACEHOLDERS = ("מישהו", "משהו", "מישהי", "כלשהו", "כלשהי")   # QASem's SRL argument placeholders


def _stems(text: str) -> set:
    """3-letter Hebrew stems of a text, for premise-grounding checks."""
    out = set()
    for w in text.split():
        h = "".join(HEB.findall(w))
        if len(h) >= 3:
            out.add(h[:3])
        elif w.strip(".,;:()\"'").isdigit():
            out.add(w.strip(".,;:()\"'"))
    return out


def grounded(entailment: str, cand: dict, premise_stems: set) -> bool:
    """A contradiction has to contradict something the premise actually says.
    The rules edit a token of the hypothesis without consulting the premise, so
    9-34% of shipped contradictions changed a detail the premise never asserted; those cannot be contradictions at all. Keep a
    candidate only when what it REPLACED is grounded in the premise."""
    import difflib
    a, b = entailment.split(), cand["text"].split()
    orig = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            orig += a[i1:i2]
    if not orig:
        return False
    hit = 0
    for w in orig:
        h = "".join(HEB.findall(w))
        key = h[:3] if len(h) >= 3 else w.strip(".,;:()\"'")
        if key and key in premise_stems:
            hit += 1
    return hit > 0


def load_entailments(src: str):
    rows = []
    for sp in ("train", "dev", "test"):
        for l in open(STAGE1 / src / f"{sp}.jsonl", encoding="utf-8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r["label"] == "entailment":
                rows.append({"item_id": f"{src}/{sp}/{r.get('id', len(rows))}", "src": src, "split": sp,
                             "v2b_id": r.get("id"), "premise": r["premise"], "entailment": r["hypothesis"]})
    return rows


def candidates_for(hyp: str, premise: str, pool: dict, rng) -> list:
    sents = [s.strip() for s in cu._SENT_SPLIT.split(hyp) if s.strip()]
    cands = []
    for si, sent in enumerate(sents):
        if len(sent.split()) < 4 and len(sents) > 1:
            continue
        for text1, name1 in G.all_candidates(sent, premise, pool, rng):
            full = list(sents); full[si] = text1; full1 = " ".join(full)
            cands.append({"text": full1, "rules": [name1], "sentence": si, **G.edit_info(hyp, full1)})
    for c in list(cands):
        if sum(1 for x in cands if len(x["rules"]) > 1) >= MAX_COMPOUND_PER_ITEM:
            break
        si = c["sentence"]; sents1 = [s.strip() for s in cu._SENT_SPLIT.split(c["text"]) if s.strip()]
        if si >= len(sents1): continue
        fam1 = G.family(c["rules"][0])
        allowed = {n for n in G.FAMILY if G.FAMILY[n] != fam1} | ({"number_plausible"} if fam1 != "number" else set()) \
                  | ({"entity_swap_v3_per", "entity_swap_v3_gpe", "entity_swap_v3_org", "entity_swap_v3_timex"} if fam1 != "entity" else set())
        first_span = set(G.edit_info(hyp, c["text"])["changed_orig_idx"])
        for text2, name2 in G.all_candidates(sents1[si], premise, pool, rng, allowed=allowed):
            if G.family(name2) == fam1: continue
            full = list(sents1); full[si] = text2; full2 = " ".join(full)
            e2 = G.edit_info(hyp, full2)
            if set(G.edit_info(c["text"], full2)["changed_orig_idx"]) & first_span: continue
            if e2["n_edit_tokens"] < 2 or full2 == c["text"]: continue
            cands.append({"text": full2, "rules": c["rules"] + [name2], "sentence": si, **e2}); break
    seen, uniq = set(), []
    for c in cands:
        if c["text"] not in seen and c["text"] != hyp:
            seen.add(c["text"]); uniq.append(c)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--shard", default="", help="I/N over entailment rows")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-grounding", action="store_true", help="skip the premise-grounding filter (rule edits are kept without checking the premise)")
    ap.add_argument("--keep-placeholders", action="store_true", help="keep QASem hypotheses containing SRL placeholders")
    args = ap.parse_args()
    out_dir = OUT / args.src; out_dir.mkdir(parents=True, exist_ok=True)
    pool = json.load(open(RESOURCES / args.src / "entity_pool.json", encoding="utf-8"))
    rows = load_entailments(args.src)
    if args.limit: rows = rows[:args.limit]
    sfx = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/")); rows = rows[i::n]; sfx = f".shard{i}of{n}"
    t0 = time.time(); n_with = 0; n_comp = 0; n_ungrounded = 0; n_placeholder = 0; by_rule = defaultdict(int)
    print(f"[{args.src}{sfx}] {len(rows)} entailment rows; pool { {k: len(v) for k, v in pool.items()} }", flush=True)
    with open(out_dir / f"candidates{sfx}.jsonl", "w", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            if not args.keep_placeholders and any(ph in r["entailment"] for ph in PLACEHOLDERS):
                n_placeholder += 1
                f.write(json.dumps({**r, "candidates": []}, ensure_ascii=False) + "\n")
                continue
            rng = G.RecordingRNG(G.SEED + k)
            c = candidates_for(r["entailment"], r["premise"], pool, rng)
            if not args.no_grounding:
                ps = _stems(r["premise"])
                before = len(c)
                c = [x for x in c if grounded(r["entailment"], x, ps)]
                n_ungrounded += before - len(c)
            n_with += bool(c); n_comp += sum(1 for x in c if len(x["rules"]) > 1)
            for x in c: by_rule["+".join(x["rules"])] += 1
            f.write(json.dumps({**r, "candidates": c}, ensure_ascii=False) + "\n")
            if (k + 1) % 1000 == 0:
                print(f"[{args.src}{sfx}] {k+1}/{len(rows)} rows, {n_with} with a candidate ({time.time()-t0:.0f}s)", flush=True)
    summary = {"src": args.src, "shard": args.shard, "rows": len(rows), "rows_with_candidate": n_with,
               "candidates_dropped_ungrounded": n_ungrounded, "rows_skipped_placeholder": n_placeholder,
               "compound_candidates": n_comp, "candidates_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
               "elapsed_s": round(time.time() - t0)}
    json.dump(summary, open(out_dir / f"candidates_summary{sfx}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates_by_rule"})); print("DONE", flush=True)


if __name__ == "__main__":
    main()
