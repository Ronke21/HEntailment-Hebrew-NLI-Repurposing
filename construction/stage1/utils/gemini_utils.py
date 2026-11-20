"""OpenRouter (google/gemini-3.1-flash-lite) batching with JSON caching, tqdm progress, and retry logic."""

import json
import os
import time
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_NAME = "google/gemini-3.1-flash-lite"

HYPOTHESIS_PROMPT = (
    "המשפט הבא הוא שאלה ותשובה בעברית. "
    "כתוב משפט הצהרתי אחד בעברית תקנית שמבטא את התשובה לשאלה. "
    "החזר רק את המשפט, ללא הסבר.\n"
    "שאלה: {question}\n"
    "תשובה: {answer}"
)

TITLE_PROMPT = (
    "קרא את הכתבה הבאה וכתוב כותרת קצרה בעברית תקנית (עד 20 מילים) "
    "שמסכמת את הנושא המרכזי. החזר רק את הכותרת, ללא הסבר.\n\n"
    "כתבה:\n{article}"
)

DESEG_PROMPT = (
    "הטקסט הבא הוא משפט עברי מפוצל מורפולוגית (כל מורפמה מופרדת ברווח). "
    "חבר אותו לטקסט עברי רגיל ורצוף ללא פיצול. "
    "החזר רק את המשפט המחובר, ללא הסבר.\n\n"
    "משפט: {sentence}"
)


def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def batch_generate(
    items: list[dict],
    prompt_template: str,
    cache_path: Path,
    batch_size: int = 100,
    sleep_between_batches: float = 1.0,
    max_retries: int = 3,
) -> list[str]:
    """
    Generate text for each item via OpenRouter. Results cached to cache_path (JSON).
    Re-running is safe — cached items are never re-called.
    """
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info(f"Cache loaded: {len(cache)}/{len(items)} already done ({cache_path.name})")

    uncached = [i for i in range(len(items)) if str(i) not in cache]
    if not uncached:
        logger.info("All items cached — no API calls needed.")
        return [cache.get(str(i), "") for i in range(len(items))]

    logger.info(f"OpenRouter ({MODEL_NAME}): {len(uncached)} items to generate ({len(items)-len(uncached)} cached)")
    client = None
    errors = 0

    with tqdm(total=len(uncached), desc=f"OpenRouter [{cache_path.stem}]", unit="item") as pbar:
        for batch_start in range(0, len(uncached), batch_size):
            batch_indices = uncached[batch_start : batch_start + batch_size]
            if client is None:
                client = _get_client()

            for i in batch_indices:
                prompt = prompt_template.format(**items[i])
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        cache[str(i)] = response.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.warning(f"API failed item {i} after {max_retries} attempts: {e}")
                            cache[str(i)] = ""
                            errors += 1
                        else:
                            time.sleep(2 ** attempt)
                pbar.update(1)

            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            if batch_start + batch_size < len(uncached):
                time.sleep(sleep_between_batches)

    logger.info(f"OpenRouter done. Errors: {errors}/{len(uncached)}. Cache saved to {cache_path}")
    return [cache.get(str(i), "") for i in range(len(items))]
