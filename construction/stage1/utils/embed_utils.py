"""dicta-il/neodictabert-bilingual-embed sentence embedding and neutral-pair selection."""

from __future__ import annotations   # PEP 604 (`X | None`) under Python 3.9

import logging
import os
import numpy as np
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_NAME = "dicta-il/neodictabert-bilingual-embed"
_model = None


def _load_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        if torch.cuda.is_available():
            _model = _model.to(torch.device("cuda"))
            dev = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(dev) // 1024 ** 2
            logger.info(f"{MODEL_NAME} loaded on GPU:{dev} | mem used: {used} MB")
        else:
            logger.warning("CUDA not available — running on CPU (slow)")
    return _model


def encode(texts: list[str], cache_path: "Path | None" = None, batch_size: int = 64) -> np.ndarray:
    """
    Encode texts with neodictabert-bilingual-embed. Loads from cache if it exists and size matches.
    Returns L2-normalized embeddings (cosine similarity = dot product).
    """
    if cache_path and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            logger.info(f"Embeddings loaded from cache: {cache_path} ({arr.shape})")
            return arr
        logger.warning(
            f"Cache size mismatch ({arr.shape[0]} vs {len(texts)} texts) — re-encoding. "
            f"Delete {cache_path} to suppress this warning."
        )

    model = _load_model()
    logger.info(f"Encoding {len(texts):,} texts with {MODEL_NAME} ...")
    result = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        np.save(cache_path, result)
        logger.info(f"Embeddings saved → {cache_path} ({result.shape})")

    return result


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def select_neutral_pairs(
    embeddings: np.ndarray,
    doc_ids: list,
    min_dist_rank: int = 0,
    max_dist_rank: int = 200,
    max_cosine_sim: float = 0.7,
    max_jaccard: float = 0.4,
    texts: list[str] | None = None,
    random_state: int = 42,
    hard_negatives: bool = False,
    hard_neg_pool: int = 50,
    min_cosine_sim: float | None = None,
    min_jaccard: float = 0.0,
) -> list[tuple[int, int]]:
    """
    For each row i, sample one neutral partner j.

    Two recipes, selected by `min_cosine_sim`:

    v1 — ORIGINAL / "most-distant" (min_cosine_sim=None, the default).
        Candidates are ordered most-distant-first and the window
        [min_dist_rank:max_dist_rank] is taken from the FRONT, i.e. the ~200
        LEAST similar rows in the split. Kept for reproducibility of the
        original `repurposed_data*` datasets.

        KNOWN DEFECT: this makes `neutral` pairs topically
        disjoint from their premise, which is trivially detectable. A 5-feature
        surface classifier scores 0.71-0.98 macro-F1 on the resulting datasets
        vs ~0.33 (random) on real Hebrew NLI, and models fine-tuned on them
        stop predicting `neutral` almost entirely on real NLI (recall 0.00-0.03).
       

    v2 — HARD NEGATIVES (hard_negatives=True).
        Takes the `hard_neg_pool` candidates that are the MOST similar while
        still under `max_cosine_sim` — i.e. `candidates[-hard_neg_pool:]`,
        the BACK of the same ascending list v1 reads from the front. These are
        topically RELATED to the premise but not close enough to entail it.
        This is what real NLI `neutral` looks like (premise/hypothesis lexical
        overlap 0.19-0.35, vs 0.02-0.16 under v1).

        Deliberately rank-based, NOT an absolute cosine floor: measured on the
        cached embeddings, median pairwise cosine differs ~6x across datasets
        (hesum p50=0.406 vs heb_paraphrase p50=0.070). A fixed floor of 0.40
        leaves 23%/47%/41% of heq/qasem/heb_paraphrase rows with an EMPTY band,
        which silently falls back to v1 pairs and reintroduces the artifact for
        half the data. Rank-from-the-top self-calibrates per dataset.

        `min_cosine_sim` / `min_jaccard` remain available as optional extra
        floors, but default to off for exactly the reason above.

    Common filtering (both recipes):
    - doc_ids[j] != doc_ids[i]                    — different source document
    - cosine_sim(i,j) <= max_cosine_sim           — not close enough to entail
    - jaccard <= max_jaccard                      — not near-duplicate (if texts given)
    """
    rng = np.random.default_rng(random_state)
    sim = embeddings @ embeddings.T  # cosine sim (embeddings are L2-normalized)

    # Recipe switch. Env var lets the whole 5-converter pipeline be flipped from
    # the regeneration command (NEUTRAL_RECIPE=v2 ./run_full_all.sh ...) without
    # threading a flag through five separate argparsers — but an explicit
    # hard_negatives=True argument always wins. The chosen recipe is logged
    # below on every call so it is never ambiguous which one produced a dataset.
    hard_neg = hard_negatives or os.environ.get("NEUTRAL_RECIPE", "").lower() in ("v2", "hard", "hard_negatives")
    pairs = []
    fallback_count = 0
    band_empty_count = 0

    for i in tqdm(range(len(embeddings)), desc="Selecting neutral pairs", unit="row"):
        row_sim = sim[i]
        sorted_j = np.argsort(row_sim)  # ascending = most distant first

        candidates = []
        for j_raw in sorted_j:
            j = int(j_raw)
            if j == i:
                continue
            cs = float(row_sim[j])
            if cs > max_cosine_sim:
                break  # sorted ascending: all remaining are even more similar
            if min_cosine_sim is not None and cs < min_cosine_sim:
                continue  # below the optional floor — too unrelated (the v1 defect)
            if doc_ids[j] == doc_ids[i]:
                continue
            if texts is not None:
                jac = _jaccard(texts[i], texts[j])
                if jac > max_jaccard:
                    continue          # near-duplicate → risks being an entailment
                if hard_neg and jac < min_jaccard:
                    continue          # too lexically disjoint → the v1 artifact
            candidates.append(j)

        if hard_neg:
            # Back of the ascending list = MOST similar under the ceiling, i.e.
            # the hardest negatives. v1 reads the front of this same list.
            windowed = candidates[-hard_neg_pool:]
            if not windowed:
                band_empty_count += 1
        else:
            windowed = candidates[min_dist_rank:max_dist_rank]
            if not windowed:
                windowed = candidates[:max_dist_rank] if candidates else []

        if not windowed:
            # Last-resort fallback: any row from a different doc, ignoring similarity ceiling
            windowed = [int(j) for j in sorted_j if int(j) != i and doc_ids[int(j)] != doc_ids[i]][:10]
            if not windowed:
                windowed = [int(j) for j in sorted_j if int(j) != i][:5]
            if windowed:
                fallback_count += 1

        if windowed:
            pairs.append((i, int(rng.choice(windowed))))
        else:
            logger.warning(f"No valid neutral candidate for row {i}")

    recipe = (f"v2/hard-negatives (top-{hard_neg_pool} most-similar under cos<={max_cosine_sim}"
              + (f", cos floor {min_cosine_sim}" if min_cosine_sim is not None else "")
              + (f", jaccard floor {min_jaccard}" if min_jaccard else "") + ")") if hard_neg else \
             f"v1/most-distant (rank window [{min_dist_rank}:{max_dist_rank}])"
    logger.info(f"Neutral selection recipe: {recipe}")
    if band_empty_count:
        logger.warning(
            f"Hard-negative pool was EMPTY for {band_empty_count}/{len(embeddings)} rows "
            f"({100*band_empty_count/max(1,len(embeddings)):.1f}%) — these fell back to "
            f"relaxed (v1-style) selection and reintroduce the artifact. If this is "
            f"non-trivial, loosen min_cosine_sim/min_jaccard.")
    if fallback_count:
        logger.warning(f"Fallback neutral selection used for {fallback_count} rows (relaxed filters)")
    logger.info(f"Neutral pairs selected: {len(pairs)}/{len(embeddings)}")
    return pairs
