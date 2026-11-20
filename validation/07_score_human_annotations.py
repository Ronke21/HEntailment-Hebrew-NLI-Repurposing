#!/usr/bin/env python
"""
07_score_human_annotations.py -- score the human validation study.

Reads the two returned annotation workbooks, maps the Hebrew labels back to the
English label space, and reports:

  1. inter-annotator agreement on the items BOTH annotated (raw + Cohen's kappa)
  2. each annotator against the pipeline's automatic label, overall and per
     source and per label
  3. the same restricted to items both annotated and agreed on -- the closest
     thing to adjudicated gold available without a third pass
  4. the ill-formed rate, i.e. how often an annotator judged the item itself
     malformed rather than assigning one of the three NLI labels

The fourth annotation category matters. Annotators could mark an item
`לא-תקין` (ill-formed) instead of choosing a label. That is not a label
disagreement: it says the item should not have been emitted at all. It is
reported separately and never silently folded into one of the three classes.

USAGE
    python 07_score_human_annotations.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FROM = HERE / "sampled_data" / "from_annotators"
KEY = HERE / "sampled_data" / "blind_key.csv"
OUT = HERE / "human_validation_results"

HE2EN = {"היסק": "entailment", "סתירה": "contradiction", "ניטרלי": "neutral"}
ILLFORMED = "לא-תקין"
LABELS = ["entailment", "contradiction", "neutral"]


def read_xlsx(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    hdr = [str(h).strip() if h else "" for h in rows[0]]
    out = []
    for r in rows[1:]:
        d = {hdr[i]: r[i] for i in range(len(hdr)) if hdr[i]}
        if d.get("item_id"):
            out.append(d)
    return out


def norm(v):
    """-> english label, 'ill_formed', or None if unannotated."""
    if v is None:
        return None
    t = str(v).strip()
    if not t:
        return None
    if t == ILLFORMED or "תקין" in t:
        return "ill_formed"
    return HE2EN.get(t, t.lower())


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def agree_block(pairs, title):
    """pairs: list of (gold, human). Returns dict; prints a block."""
    if not pairs:
        print(f"  {title}: no items"); return {}
    g = [x for x, _ in pairs]; h = [y for _, y in pairs]
    acc = sum(x == y for x, y in pairs) / len(pairs)
    print(f"  {title:44s} n={len(pairs):4d}  agree={acc:.3f}  kappa={kappa(g, h):.3f}")
    return dict(n=len(pairs), agreement=acc, kappa=kappa(g, h))


def main():
    key = {}
    import csv
    with open(KEY, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            key[r["item_id"]] = r

    files = sorted(FROM.glob("*.xlsx"))
    ann = {}
    for f in files:
        name = "A1" if "annotator_1" in f.name else "A2"
        rows = read_xlsx(f)
        d = {}
        for r in rows:
            lab = norm(r.get("annotator_label"))
            new = norm(r.get("new label"))
            if lab is not None:
                d[r["item_id"]] = dict(label=lab, new_label=new)
        ann[name] = d
        n_ill = sum(1 for v in d.values() if v["label"] == "ill_formed")
        print(f"{name}: {len(d)}/315 annotated, {n_ill} marked ill-formed"
              f" ({n_ill/len(d):.1%}), {sum(1 for v in d.values() if v['new_label'])} proposed relabels")

    A1, A2 = ann["A1"], ann["A2"]
    both = sorted(set(A1) & set(A2))
    print(f"\ndouble-annotated overlap: {len(both)} items")

    res = {}
    print("\n=== 1. inter-annotator agreement (overlap only) ===")
    pairs = [(A1[i]["label"], A2[i]["label"]) for i in both]
    res["iaa_all4"] = agree_block(pairs, "all four categories")
    pairs3 = [(A1[i]["label"], A2[i]["label"]) for i in both
              if A1[i]["label"] != "ill_formed" and A2[i]["label"] != "ill_formed"]
    res["iaa_3way"] = agree_block(pairs3, "three NLI labels only (ill-formed dropped)")

    print("\n=== 2. annotator vs pipeline label ===")
    for name, d in ann.items():
        p = [(key[i]["label"], d[i]["label"]) for i in d if i in key]
        res[f"{name}_vs_pipeline_all4"] = agree_block(p, f"{name}: all four categories")
        p3 = [(key[i]["label"], d[i]["label"]) for i in d
              if i in key and d[i]["label"] != "ill_formed"]
        res[f"{name}_vs_pipeline_3way"] = agree_block(p3, f"{name}: ill-formed dropped")

    print("\n=== 3. pipeline vs the double-annotated consensus ===")
    cons = [(key[i]["label"], A1[i]["label"]) for i in both
            if A1[i]["label"] == A2[i]["label"]]
    res["pipeline_vs_consensus"] = agree_block(cons, "both annotators agreed on these")

    print("\n=== 4. per source (pipeline agreement, ill-formed counted as disagreement) ===")
    for name, d in ann.items():
        per = defaultdict(lambda: [0, 0])
        for i in d:
            if i not in key: continue
            src = key[i]["dataset"]
            per[src][1] += 1
            per[src][0] += int(d[i]["label"] == key[i]["label"])
        res[f"{name}_per_source"] = {k: dict(agree=v[0]/v[1], n=v[1]) for k, v in per.items()}
        print(f"  {name}: " + "  ".join(f"{k}={v[0]/v[1]:.3f}(n={v[1]})" for k, v in sorted(per.items())))

    print("\n=== 5. per gold label ===")
    for name, d in ann.items():
        per = defaultdict(lambda: [0, 0, 0])   # agree, n, ill-formed
        for i in d:
            if i not in key: continue
            g = key[i]["label"]
            per[g][1] += 1
            per[g][0] += int(d[i]["label"] == g)
            per[g][2] += int(d[i]["label"] == "ill_formed")
        res[f"{name}_per_label"] = {k: dict(agree=v[0]/v[1], n=v[1], ill_formed=v[2]) for k, v in per.items()}
        print(f"  {name}: " + "  ".join(
            f"{k}={v[0]/v[1]:.3f}(ill={v[2]})" for k, v in sorted(per.items())))

    print("\n=== 6. where the humans put the pipeline's label instead ===")
    for name, d in ann.items():
        conf = Counter((key[i]["label"], d[i]["label"]) for i in d
                       if i in key and d[i]["label"] != key[i]["label"])
        print(f"  {name}: " + ", ".join(f"{g}->{h}:{c}" for (g, h), c in conf.most_common(6)))

    OUT.mkdir(exist_ok=True)
    (OUT / "human_validation_summary.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT/'human_validation_summary.json'}")


if __name__ == "__main__":
    main()
