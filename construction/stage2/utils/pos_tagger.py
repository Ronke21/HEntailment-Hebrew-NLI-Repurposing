"""
Thin wrapper around dicta-il/dictabert-joint (POS, morphology and NER for
Hebrew), used by two rules:

  - verb negation: negates only a token the tagger calls VERB, and does not
    fire on a sentence without one;
  - entity swap: gated on the tagger's NER, so a PER is only swapped for a
    different PER, a GPE for a different GPE, and so on.

API note: BertForJointParsing.predict() checks
`if compute_syntax_mst is not None and self.syntax is None: raise`, so the
syntax head must be skipped with compute_syntax_mst=None, not False.
"""

from __future__ import annotations
import os
import re
import random as _random
from functools import lru_cache
from typing import Optional

_MODEL = None
_TOKENIZER = None


def _load():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        # Without this, a fresh process re-checks the Hub for updates before
        # loading an already-fully-cached model -- normally fast, but on one
        # occasion during this build it hung for >2 minutes with no error
        # (verified: setting this dropped load time to a consistent ~50s).
        # Caching happens once; querying the Hub on every process start for
        # a model that's already local is unnecessary regardless.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModel, AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint")
        _MODEL = AutoModel.from_pretrained(
            "dicta-il/dictabert-joint", trust_remote_code=True,
            do_syntax=False, do_lex=False,
        )
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def tag_batch(sentences: list[str]) -> list[dict]:
    """Returns dictabert-joint's JSON output: per-sentence tokens (with
    morph.pos / morph.feats) and ner_entities (phrase, label, char span)."""
    if not sentences:
        return []
    model, tok = _load()
    # See module docstring: compute_syntax_mst=None, NOT False.
    return model.predict(sentences, tok, output_style="json", compute_syntax_mst=None)


def tag_one(sentence: str) -> dict:
    return tag_batch([sentence])[0]


# ── B2 fix: POS-gated verb negation (replaces Heuristic 13) ─────────────────

def find_finite_verb_span(text: str) -> Optional[tuple[int, int, str]]:
    """Returns (char_start, char_end, surface_form) of the first token tagged
    VERB, or None if the sentence has no finite verb -- in which case the
    caller must not fire verb-negation at all (this is the fix: the old
    heuristic always found *something* to negate; a sentence like "בירת יפן
    היא טוקיו" has no verb and should be left alone)."""
    r = tag_one(text)
    offset = 0
    for t in r["tokens"]:
        if t["morph"]["pos"] == "VERB":
            start = text.find(t["token"], offset)
            if start == -1:
                continue
            return start, start + len(t["token"]), t["token"]
        offset += len(t["token"])
    return None


def try_verb_negation_v3(text: str) -> Optional[tuple[str, str]]:
    """POS-gated replacement for the old _try_verb_negation. Inserts 'לא'
    immediately before the sentence's first real VERB token. Returns None
    (no contradiction produced) if there is no finite verb -- correct and
    deliberate: not every sentence can be turned into a verb-negated
    contradiction, and guessing was the bug."""
    if "לא" in text.split():
        return None
    span = find_finite_verb_span(text)
    if span is None:
        return None
    start, _end, _verb = span
    out = text[:start] + "לא " + text[start:]
    return out, "verb_negation_v3"



_SWAPPABLE_TYPES = {"PER", "GPE", "ORG", "TIMEX"}


def _entity_core_and_prefix(tokens: list[dict], token_start: int, token_end: int) -> tuple[str, str]:
    """For a multi-token NER span, split off the first token's prefix (if
    any) and return (surface_form_without_that_prefix, prefix_string)."""
    span_tokens = tokens[token_start : token_end + 1]
    first = span_tokens[0]
    seg = first.get("seg") or (first["token"],)
    prefixes = first["morph"].get("prefixes") or []
    if prefixes and len(seg) > 1:
        prefix_str = seg[0]
        first_core = "".join(seg[1:])
    else:
        prefix_str = ""
        first_core = first["token"]
    core_tokens = [first_core] + [t["token"] for t in span_tokens[1:]]
    return " ".join(core_tokens), prefix_str


def try_entity_swap_v3(
    text: str, entity_pool: dict[str, list[str]], rng: Optional[_random.Random] = None
) -> Optional[tuple[str, str]]:
    """Swap one NER-tagged entity mention in `text` for a different
    same-type entity drawn from `entity_pool` (label -> de-prefixed core
    forms seen elsewhere in the corpus), reattaching THIS token's own
    prefix so the preposition stays grammatical. None if the sentence has
    no swappable entity or the pool has no alternative for its type.

    KNOWN LIMITATION, not fixed: does not re-inflect the sentence's verb
    for the replacement entity's gender/number (e.g. a masculine verb stays
    masculine even when swapped onto a feminine PER). Left for the D3
    grammaticality judge to catch downstream, same backstop the LLM-
    generated (D1) contradictions rely on -- full agreement correction
    would need generation, not substitution, to do properly."""
    if rng is None:
        rng = _random.Random()
    r = tag_one(text)
    ents = [e for e in r["ner_entities"] if e["label"] in _SWAPPABLE_TYPES]
    if not ents:
        return None
    rng.shuffle(ents)
    for e in ents:
        core, prefix = _entity_core_and_prefix(r["tokens"], e["token_start"], e["token_end"])
        pool = [p for p in entity_pool.get(e["label"], []) if p != core]
        if not pool:
            continue
        replacement = prefix + rng.choice(pool)
        out = text[: e["start"]] + replacement + text[e["end"] :]
        return out, f"entity_swap_v3_{e['label'].lower()}"
    return None


def build_entity_pool(premises: list[str], min_count: int = 3) -> dict[str, list[str]]:
    """Tag a batch of premises and collect a label -> [de-prefixed core
    surface forms] pool for try_entity_swap_v3. Run once at corpus-build
    time, cache to disk."""
    pool: dict[str, set[str]] = {}
    BATCH = 32
    for i in range(0, len(premises), BATCH):
        for r in tag_batch(premises[i : i + BATCH]):
            for e in r["ner_entities"]:
                if e["label"] in _SWAPPABLE_TYPES:
                    core, _prefix = _entity_core_and_prefix(r["tokens"], e["token_start"], e["token_end"])
                    pool.setdefault(e["label"], set()).add(core)
    return {k: sorted(v) for k, v in pool.items() if len(v) >= min_count}
