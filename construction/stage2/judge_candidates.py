#!/usr/bin/env python3
"""
Judge prompts, label parsing, rule exclusion and candidate ranking shared by
the stage-2 scripts.

GRAMMAR_PROMPT asks whether a Hebrew sentence is well-formed; LABEL_PROMPT asks
for the relation between a passage and a sentence (entailment / contradiction /
neutral). Both are used verbatim with gemma-4-31B-it as the construction judge
and with Qwen3.6-27B as the independent second judge (quality/).
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

REPO = __import__("pathlib").Path(__import__("os").environ.get("HENT_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = REPO / "construction/resources"
LABELS = ("entailment", "contradiction", "neutral")
LABEL_PROMPT = """קרא את הפסקה והמשפט הבא. מהו היחס הלוגי בין הפסקה למשפט?
ענה במילה אחת בלבד מתוך: entailment, contradiction, neutral.

פסקה:
{premise}

משפט:
{hypothesis}
"""
GRAMMAR_PROMPT = """המשימה שלך: לבדוק אם המשפט הבא תקין מבחינה **דקדוקית ותחבירית** בעברית -
כלומר, האם הוא בנוי נכון כמשפט (לא אם התוכן שלו נכון עובדתית או סביר).
משפט תקין דקדוקית יכול להכיל טענה מוזרה, שגויה או בלתי סבירה - זה לא משנה את התשובה.
פסול רק אם יש שגיאת ניסוח, מילה שבורה, או מבנה תחבירי שגוי שהופך את המשפט לבלתי קריא.

ענה במילה אחת בלבד: תקין / לא-תקין

משפט: {text}
"""
MAX_PREMISE_WORDS = 2000


MARKER_FAMILIES = {"negation", "quantity", "modality"}
FAMILY = {"number_substitution": "number", "number_substitution_heb": "number", "number_plausible": "number",
          "verb_negation_v3": "negation", "existential_negation": "negation", "quantifier_flip": "quantity",
          "modal_flip": "modality", "epistemic_certainty": "modality", "temporal_inversion": "time",
          "superlative_flip": "degree", "scope_severity": "degree", "pace_velocity": "degree",
          "adjective_antonym": "lexical", "causal_reversal": "lexical", "geo_directional_flip": "lexical",
          "decision_inversion": "outcome", "financial_outcome": "outcome", "legal_status": "outcome",
          "inclusion_exclusion": "outcome", "factional_alliance": "outcome", "state_transition": "outcome"}


def _fam(n): return "entity" if n.startswith("entity_swap") else FAMILY.get(n, n)


def load_excluded(d) -> set:
    """The rules a source excludes from the corpus because they leak the label read
    from <src>/exclude_rules.txt."""
    from pathlib import Path as _P
    f = _P(d) / "exclude_rules.txt"
    return {r.strip() for r in f.read_text(encoding="utf-8").split() if r.strip()} if f.exists() else set()


def eligible(cands, excluded):
    """Candidates selection could keep: those not from an excluded rule.
    Exclusion is applied before ranking, so the judge never spends a slot on a
    candidate that selection would discard."""
    if not excluded:
        return list(cands)
    return [c for c in cands if not any(r in excluded for r in c["rules"])]


def rank_candidates(cands, k):
    """Stage C's ranking, applied BEFORE judging: >=2-token edits first, fact
    rules before marker-inserting ones, fewer inserted tokens. Judging only the
    top k per item is what makes the long-form sources affordable (HebSummaries
    averages 22 rule candidates per item, Hebrew-Paraphrase 9.6); stage C keeps
    exactly one per item anyway, and a wider excluded-rule list would only shrink
    the pool further. k=0 judges all."""
    if k <= 0 or len(cands) <= k:
        return cands
    ranked = sorted(cands, key=lambda c: (not c.get("grounded", True),
                                          c["n_edit_tokens"] < 2,
                                          any(_fam(r) in MARKER_FAMILIES for r in c["rules"]),
                                          len(c["inserted_tokens"])))
    return ranked[:k]


def parse_label(o: str):
    o = o.strip().lower()
    for lab in LABELS:
        if lab in o: return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="heq")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=8, help="judge only the top-k ranked candidates per item (0 = all)")
    ap.add_argument("--chunk", type=int, default=4000, help="texts per judged+written chunk (a 4h kill costs at most one chunk)")
    ap.add_argument("--shard", default="", help="I/N: judge only items i%N==I (each shard writes its own file; stage C merges)")
    args = ap.parse_args()
    tag = args.model.split("/")[-1].replace(".", "_")
    d = OUT / args.src
    items = [json.loads(l) for l in open(d / "candidates.jsonl", encoding="utf-8")]
    if args.limit: items = items[:args.limit]
    shard_sfx = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/")); items = items[i::n]; shard_sfx = f".shard{i}of{n}"
    # units to judge: (kind, item_id, text, label_expected)
    units = []
    n_skip = sum(1 for it in items if not it["candidates"])
    items = [it for it in items if it["candidates"]]   # an item with no rule candidate can never be kept: do not spend judge time on it
    print(f"[{args.src}] skipping {n_skip} items without candidates", flush=True)
    for it in items:
        units.append(("entailment", it["item_id"], it["entailment"], "entailment"))
        if it.get("neutral"): units.append(("neutral", it["item_id"], it["neutral"], "neutral"))
        keep = rank_candidates(it["candidates"], args.max_candidates)
        for ci, c in enumerate(it["candidates"]):
            if c in keep:
                units.append((f"cand{ci}", it["item_id"], c["text"], "contradiction"))
    prem = {it["item_id"]: " ".join(it["premise"].split()[:MAX_PREMISE_WORDS]) for it in items}
    # reuse verdicts already produced for the same (premise, text): only new candidates cost GPU time
    out_f = d / f"verdicts_{tag}{shard_sfx}.jsonl"; cache = {}
    for f0 in sorted(d.glob(f"verdicts_{tag}*.jsonl")) + sorted(d.glob(f"verdicts_{tag}*.partial.jsonl")):
        for l in open(f0, encoding="utf-8"):
            v = json.loads(l); cache[(v["item_id"], v.get("text"))] = v
    todo = [u for u in units if (u[1], u[2]) not in cache]
    print(f"[{args.src}] cached verdicts: {len(units) - len(todo)}, to judge now: {len(todo)}", flush=True)
    units_all = units; units = todo
    print(f"[{args.src}] {len(items)} items, {len(units)} texts to judge with {args.model}", flush=True)

    if not units:
        lab_out, gram = [], {}
    from vllm import LLM, SamplingParams
    import torch
    free_b, total_b = torch.cuda.mem_get_info()
    frac = max(0.30, min(0.95, free_b * 0.97 / total_b))
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=frac, max_num_seqs=args.max_num_seqs, trust_remote_code=True)
    tok = llm.get_tokenizer()
    def render(content):
        msgs = [{"role": "user", "content": content}]
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(temperature=0, max_tokens=12)
    t0 = time.time()
    part_f = d / f"verdicts_{tag}{shard_sfx}.partial.jsonl"
    with open(part_f, "a", encoding="utf-8") as pf:
        for c0 in range(0, len(units), args.chunk):
            chunk = units[c0:c0 + args.chunk]
            lab_out = llm.generate([render(LABEL_PROMPT.format(premise=prem[i], hypothesis=t)) for _, i, t, _ in chunk], sp)
            texts = sorted({t for _, _, t, _ in chunk})
            gr_out = llm.generate([render(GRAMMAR_PROMPT.format(text=t)) for t in texts], sp)
            gram = {t: o.outputs[0].text.strip() for t, o in zip(texts, gr_out)}
            for (kind, iid, text, exp), o in zip(chunk, lab_out):
                raw = o.outputs[0].text.strip(); lab = parse_label(raw); g = gram[text]
                v = {"item_id": iid, "kind": kind, "expected": exp, "text": text, "label_raw": raw, "label": lab,
                     "label_ok": lab == exp, "gram_raw": g, "gram_ok": g.startswith("תקין"), "judge": args.model}
                cache[(iid, text)] = v
                pf.write(json.dumps(v, ensure_ascii=False) + "\n")
            pf.flush(); os.fsync(pf.fileno())
            print(f"[{args.src}] chunk {c0 + len(chunk)}/{len(units)} judged ({time.time()-t0:.0f}s)", flush=True)
    with open(out_f, "w", encoding="utf-8") as f:
        for (kind, iid, text, exp) in units_all:
            v = dict(cache[(iid, text)]); v["kind"] = kind; v["expected"] = exp
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    (d / f"verdicts_{tag}{shard_sfx}.DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
