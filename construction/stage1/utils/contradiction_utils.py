"""
Contradiction heuristics for Hebrew NLI hypothesis texts.
All applicable heuristics are collected; apply_heuristics() returns one chosen randomly.

Changes vs. v1 (based on Gemini expert review):
- Dropped bare negation (grammatically unsafe in Hebrew without POS tagging).
  Replaced by verb_negation, which limits itself to short factoid sentences (2–12 tokens).
- Fixed number_substitution: skips 4-digit year-range numbers (1000–2099).
- Fixed number_substitution_heb: added gender/construct-state forms (שני, שתי).
- Fixed superlative_flip: added feminine and plural surface forms.
- Fixed temporal_inversion: added מלפני/מאחרי prefix forms (listed before shorter לפני/אחרי).
- Fixed quantifier_flip: removed כולם↔אף אחד and תמיד↔אף פעם (require
  Negative Concord verb rewrite); removed כל↔אף (too short, fires everywhere).
- Fixed adjective_antonym: removed homographs חם/קר (hot vs. father-in-law/read);
  added feminine surface forms and אופטימי↔פסימי.
- Fixed existential_negation: added gender/number forms (נמצאת, נמצאים, קיימים, ישנן).
- Fixed causal_reversal: NP prepositions (בגלל/בשל/עקב) now map to על אף/למרות,
  not to אף על פי ש (which requires a clause); clause connectives kept separate.
- Fixed geo_directional_flip: removed ambiguous עולה/יורד (also: immigrant, costs)
  and גדל/קטן (also: got smaller/shrunk).
- Fixed decision_inversion: added verb+preposition pairs (תמך ב ↔ התנגד ל) to
  avoid subcategorization mismatch; added feminine and plural inflections.
- Added 8 new heuristics (slots 15–22):
  inclusion_exclusion, financial_outcome, legal_status, epistemic_certainty,
  factional_alliance, pace_velocity, state_transition, scope_severity.
"""

import re
import os
import random as _random
from typing import Optional


# ── Heuristic 4: Superlative/comparative flip ────────────────────────────────
# Longer strings listed first so they match before their substrings.
SUPERLATIVE_FLIPS = [
    # Masculine/singular
    ("הגדול ביותר",  "הקטן ביותר"),   ("הקטן ביותר",   "הגדול ביותר"),
    # Feminine/singular
    ("הגדולה ביותר", "הקטנה ביותר"),  ("הקטנה ביותר",  "הגדולה ביותר"),
    # Masculine/plural
    ("הגדולים ביותר","הקטנים ביותר"), ("הקטנים ביותר", "הגדולים ביותר"),
    # Feminine/plural
    ("הגדולות ביותר","הקטנות ביותר"), ("הקטנות ביותר", "הגדולות ביותר"),
    ("הכי גבוהה",   "הכי נמוכה"),    ("הכי נמוכה",    "הכי גבוהה"),
    ("הכי גבוה",    "הכי נמוך"),     ("הכי נמוך",     "הכי גבוה"),
    ("הכי גדולה",   "הכי קטנה"),     ("הכי קטנה",     "הכי גדולה"),
    ("הכי גדול",    "הכי קטן"),      ("הכי קטן",      "הכי גדול"),
    ("הכי חזק",     "הכי חלש"),      ("הכי חלש",      "הכי חזק"),
    ("הגדולה",      "הקטנה"),        ("הקטנה",        "הגדולה"),
    ("הגדול",       "הקטן"),         ("הקטן",         "הגדול"),
    ("הראשונה",     "האחרונה"),      ("האחרונה",      "הראשונה"),
    ("הראשון",      "האחרון"),       ("האחרון",       "הראשון"),
    ("הצלחה",       "כישלון"),       ("כישלון",       "הצלחה"),
    ("עלייה",       "ירידה"),        ("ירידה",        "עלייה"),
    ("גידול",       "קיטון"),        ("קיטון",        "גידול"),
    ("שיפור",       "הידרדרות"),     ("הידרדרות",     "שיפור"),
]

# ── Heuristic 5: Temporal inversion ──────────────────────────────────────────
# מ-prefix forms listed before base forms to avoid partial matches.
TEMPORAL_FLIPS = [
    ("מלפני",   "מאחרי"),   ("מאחרי",   "מלפני"),
    ("לאחר ש",  "טרם"),     ("טרם",     "לאחר ש"),
    ("לפני",    "אחרי"),    ("אחרי",    "לפני"),
    ("בעבר",    "בעתיד"),   ("בעתיד",   "בעבר"),
    ("ראשית",   "לבסוף"),   ("לבסוף",   "ראשית"),
    ("פתיחה",   "סגירה"),   ("סגירה",   "פתיחה"),
    ("אחר כך",  "קודם"),    ("קודם",    "אחר כך"),
    ("בתחילה",  "בסוף"),    ("בסוף",    "בתחילה"),
    ("מוקדם",   "מאוחר"),   ("מאוחר",   "מוקדם"),
]

# ── Heuristic 2: Hebrew number words → replacement ───────────────────────────
# Includes gendered and construct-state forms (אחד/אחת, שני/שתי, etc.).
HEBREW_NUMBER_MAP = {
    "אחד": "עשרה",   "אחת": "עשר",
    "שניים": "עשרים", "שתיים": "עשרים",
    "שני": "עשרה",   "שתי": "עשרות",
    "שלושה": "שלושים", "שלוש": "שלושים",
    "ארבעה": "ארבעים", "ארבע": "ארבעים",
    "חמישה": "חמישים", "חמש": "חמישים",
    "שישה": "שישים",  "שש": "שישים",
    "שבעה": "שבעים",  "שבע": "שבעים",
    "שמונה": "שמונים",
    "תשעה": "תשעים",  "תשע": "תשעים",
    "עשרה": "מאה",    "עשר": "מאה",
    "עשרים": "מאתיים",
    "שלושים": "שלוש מאות",
    "מאה": "אלף",
    "אלף": "עשרת אלפים",
    "מיליון": "מיליארד",
    "מיליארד": "טריליון",
}

# ── Heuristic 6: Quantifier flip ─────────────────────────────────────────────
# Removed: כולם↔אף אחד, תמיד↔אף פעם (require Negative Concord verb rewrite),
#           כל↔אף (too short, fires on almost every sentence).
QUANTIFIER_FLIPS = [
    ("הכל",  "כלום"),   ("כלום",  "הכל"),
    ("הרבה", "מעט"),    ("מעט",   "הרבה"),
    ("רוב",  "מיעוט"),  ("מיעוט", "רוב"),
    ("רבים", "מעטים"),  ("מעטים", "רבים"),
    ("רבות", "מעטות"),  ("מעטות", "רבות"),
]

# ── Heuristic 7: Adjective antonym pairs ─────────────────────────────────────
# Removed: חם↔קר (homographs: hot/father-in-law, cold/read(past)).
# Added: feminine surface forms for top pairs; אופטימי↔פסימי.
ADJECTIVE_ANTONYMS = [
    ("חיובית",   "שלילית"),    ("שלילית",    "חיובית"),
    ("חיוביים",  "שליליים"),   ("שליליים",   "חיוביים"),
    ("חיובי",    "שלילי"),     ("שלילי",     "חיובי"),
    ("מהירה",    "איטית"),     ("איטית",     "מהירה"),
    ("מהיר",     "איטי"),      ("איטי",      "מהיר"),
    ("חדשה",     "ישנה"),      ("ישנה",      "חדשה"),
    ("חדש",      "ישן"),       ("ישן",       "חדש"),
    ("גבוהה",    "נמוכה"),     ("נמוכה",     "גבוהה"),
    ("גבוה",     "נמוך"),      ("נמוך",      "גבוה"),
    ("ארוכה",    "קצרה"),      ("קצרה",      "ארוכה"),
    ("ארוך",     "קצר"),       ("קצר",       "ארוך"),
    ("כבדה",     "קלה"),       ("קלה",       "כבדה"),
    ("כבד",      "קל"),        ("קל",        "כבד"),
    ("עשירה",    "עניה"),      ("עניה",      "עשירה"),
    ("עשיר",     "עני"),       ("עני",       "עשיר"),
    ("מוצלחת",   "כושלת"),     ("כושלת",     "מוצלחת"),
    ("מוצלח",    "כושל"),      ("כושל",      "מוצלח"),
    ("יעילה",    "בלתי יעילה"),
    ("יעיל",     "בלתי יעיל"),
    ("בטוחה",    "מסוכנת"),    ("מסוכנת",    "בטוחה"),
    ("בטוח",     "מסוכן"),     ("מסוכן",     "בטוח"),
    ("חוקית",    "בלתי חוקית"),
    ("חוקי",     "בלתי חוקי"),
    ("זמנית",    "קבועה"),     ("קבועה",     "זמנית"),
    ("זמני",     "קבוע"),      ("קבוע",      "זמני"),
    ("פומבית",   "פרטית"),     ("פרטית",     "פומבית"),
    ("פומבי",    "פרטי"),      ("פרטי",      "פומבי"),
    ("רשמית",    "לא רשמית"),
    ("רשמי",     "לא רשמי"),
    ("שלמה",     "חלקית"),     ("חלקית",     "שלמה"),
    ("שלם",      "חלקי"),      ("חלקי",      "שלם"),
    ("אופטימית", "פסימית"),    ("פסימית",    "אופטימית"),
    ("אופטימי",  "פסימי"),     ("פסימי",     "אופטימי"),
]

# ── Heuristic 8: Existential negation ────────────────────────────────────────
# Expanded with gender and number forms. Longer/more specific forms listed first.
EXISTENTIAL_PAIRS = [
    ("ישנם ",    "אינם "),
    ("ישנן ",    "אינן "),
    ("ישנה ",    "אינה "),
    ("יש ",      "אין "),
    ("קיימים ",  "אינם קיימים "),
    ("קיימות ",  "אינן קיימות "),
    ("קיימת ",   "אינה קיימת "),
    ("קיים ",    "אינו קיים "),
    ("נמצאים ",  "אינם נמצאים "),
    ("נמצאות ",  "אינן נמצאות "),
    ("נמצאת ",   "אינה נמצאת "),
    ("נמצא ",    "אינו נמצא "),
]

# ── Heuristic 9: Causal reversal ─────────────────────────────────────────────
# Grouped by subcategorization to avoid syntax errors:
#   NP prepositions (בגלל/בשל/עקב/לאור) → על אף / למרות (also take NPs).
#   Clause connectives (לכן/לפיכך/משום ש/כיוון ש) → למרות זאת / אף על פי ש.
CAUSAL_FLIPS = [
    # NP-taking prepositions
    ("בגלל ",       "על אף "),       ("על אף ",       "בגלל "),
    ("בשל ",        "למרות "),       ("למרות ",       "בשל "),
    ("עקב ",        "על אף "),
    ("לאור ",       "על אף "),
    # Clause connectives
    ("למרות זאת ",  "לכן "),
    ("לכן ",        "למרות זאת "),
    ("לפיכך ",      "למרות זאת "),
    ("כתוצאה מכך ", "למרות זאת "),
    ("משום ש",      "אף על פי ש"),   ("אף על פי ש",  "משום ש"),
    ("כיוון ש",     "אף על פי ש"),
    ("שכן ",        "אולם "),         ("אולם ",        "שכן "),
]

# ── Heuristic 10: Modal flip ─────────────────────────────────────────────────
MODAL_FLIPS = [
    ("חייב ",      "לא חייב "),
    ("מוכרח ",     "לא מוכרח "),
    ("אסור ",      "מותר "),
    ("מותר ",      "אסור "),
    ("אפשרי",      "בלתי אפשרי"),
    ("בלתי אפשרי", "אפשרי"),
    ("ניתן ",      "לא ניתן "),
    ("אפשר ",      "אי אפשר "),
    ("חובה ",      "רשות "),
    ("רשות ",      "חובה "),
]

# ── Heuristic 11: Geographic/directional flip ────────────────────────────────
# Removed עולה/יורד (ambiguous: immigrant, costs, prices) and גדל/קטן (ambiguous).
GEO_FLIPS = [
    ("צפוני",  "דרומי"),   ("דרומי",  "צפוני"),
    ("מזרחית", "מערבית"),  ("מערבית", "מזרחית"),
    ("מזרחי",  "מערבי"),   ("מערבי",  "מזרחי"),
    ("צפון",   "דרום"),    ("דרום",   "צפון"),
    ("מזרח",   "מערב"),    ("מערב",   "מזרח"),
    ("מתרחב",  "מצטמצם"),  ("מצטמצם", "מתרחב"),
    ("מתחזק",  "נחלש"),    ("נחלש",   "מתחזק"),
]

# ── Heuristic 12: Decision/approval inversion ────────────────────────────────
# תמך/התנגד now paired with their governing prepositions (ב/ל) to avoid
# subcategorization mismatch (תמך ב... vs. *התנגד ב...).
DECISION_FLIPS = [
    # Longer/preposition-bearing forms first
    ("הצביעה בעד",  "הצביעה נגד"),  ("הצביעה נגד",  "הצביעה בעד"),
    ("הצביע בעד",   "הצביע נגד"),   ("הצביע נגד",   "הצביע בעד"),
    ("התנגדו ל",    "תמכו ב"),       ("תמכו ב",      "התנגדו ל"),
    ("התנגדה ל",    "תמכה ב"),       ("תמכה ב",      "התנגדה ל"),
    ("התנגד ל",     "תמך ב"),        ("תמך ב",       "התנגד ל"),
    ("אישרה",       "דחתה"),         ("דחתה",        "אישרה"),
    ("אישרו",       "דחו"),          ("דחו",         "אישרו"),
    ("אישר",        "דחה"),          ("דחה",         "אישר"),
    ("הצליחה",      "נכשלה"),        ("נכשלה",       "הצליחה"),
    ("הצליחו",      "נכשלו"),        ("נכשלו",       "הצליחו"),
    ("הצליח",       "נכשל"),         ("נכשל",        "הצליח"),
    ("זכתה",        "הפסידה"),       ("הפסידה",      "זכתה"),
    ("זכו",         "הפסידו"),       ("הפסידו",      "זכו"),
    ("זכה",         "הפסיד"),        ("הפסיד",       "זכה"),
    ("קיבל",        "דחה"),
    ("אסר",         "התיר"),         ("התיר",        "אסר"),
    ("ביטל",        "אישר"),
    ("הוריד",       "העלה"),         ("העלה",        "הוריד"),
]

# ── Heuristic 15: Inclusion/exclusion flip ───────────────────────────────────
INCLUSION_EXCLUSION_FLIPS = [
    ("בתוספת ",  "בהפחתת "),  ("בהפחתת ",  "בתוספת "),
    ("לרבות ",   "למעט "),    ("למעט ",    "לרבות "),
    ("לרבות ",   "מלבד "),    ("מלבד ",    "לרבות "),
    ("כוללים ",  "ללא "),
    ("כוללת ",   "ללא "),
    ("כולל ",    "ללא "),     ("ללא ",     "כולל "),
]

# ── Heuristic 16: Financial outcome flip ─────────────────────────────────────
FINANCIAL_FLIPS = [
    ("הכנסות",    "הוצאות"),    ("הוצאות",    "הכנסות"),
    ("רווחים",    "הפסדים"),    ("הפסדים",    "רווחים"),
    ("גירעון",    "עודף"),      ("עודף",      "גירעון"),
    ("התכווצות",  "צמיחה"),     ("צמיחה",     "התכווצות"),
    ("פיחות",     "ייסוף"),     ("ייסוף",     "פיחות"),
    ("רווח",      "הפסד"),      ("הפסד",      "רווח"),
]

# ── Heuristic 17: Legal status flip ──────────────────────────────────────────
LEGAL_FLIPS = [
    ("הורשעו",    "זוכו"),      ("זוכו",      "הורשעו"),
    ("הורשעה",    "זוכתה"),     ("זוכתה",     "הורשעה"),
    ("הורשע",     "זוכה"),      ("זוכה",      "הורשע"),
    ("בלתי חוקית","חוקית"),
    ("בלתי חוקי", "חוקי"),      ("חוקי",      "בלתי חוקי"),
]

# ── Heuristic 18: Epistemic certainty flip ───────────────────────────────────
EPISTEMIC_FLIPS = [
    ("ללא ספק",   "מוטל בספק"),  ("מוטל בספק", "ללא ספק"),
    ("הופרך כי",  "הוכח כי"),
    ("הוכח כי",   "הופרך כי"),
    ("הופרך ש",   "הוכח ש"),
    ("הוכח ש",    "הופרך ש"),
    ("כלל לא",    "בהחלט"),
    ("בהחלט",     "כלל לא"),
    ("בוודאות",   "בספק"),       ("בספק",      "בוודאות"),
]

# ── Heuristic 19: Factional alliance flip ────────────────────────────────────
# Construct-state nouns (תומכי/מתנגדי) matched as phrases; single words
# (קואליציה/אופוזיציה) matched with word-boundary matching in apply_heuristics.
FACTIONAL_PHRASE_FLIPS = [
    ("תומכי ה",  "מתנגדי ה"),  ("מתנגדי ה",  "תומכי ה"),
    ("תומכי",    "מתנגדי"),    ("מתנגדי",    "תומכי"),
]

FACTIONAL_WORD_FLIPS = [
    ("קואליציה",  "אופוזיציה"),  ("אופוזיציה", "קואליציה"),
    ("בעד",       "נגד"),         ("נגד",        "בעד"),
]

# ── Heuristic 20: Pace/velocity flip ─────────────────────────────────────────
PACE_FLIPS = [
    ("בקצב מהיר",  "בקצב איטי"),  ("בקצב איטי",  "בקצב מהיר"),
    ("במהירות",    "באיטיות"),    ("באיטיות",    "במהירות"),
    ("הואצה",      "הואטה"),      ("הואטה",      "הואצה"),
    ("הואץ",       "הואט"),       ("הואט",       "הואץ"),
    ("מהר",        "לאט"),         ("לאט",        "מהר"),
]

# ── Heuristic 21: State transition flip ──────────────────────────────────────
# Passive binyanim (Nif'al / Huf'al) are safe for lexical swap — they rarely
# require object prepositions that would cause subcategorization mismatch.
STATE_TRANSITION_FLIPS = [
    ("נפתחו",   "נסגרו"),    ("נסגרו",    "נפתחו"),
    ("נפתחה",   "נסגרה"),    ("נסגרה",    "נפתחה"),
    ("נפתח",    "נסגר"),     ("נסגר",     "נפתח"),
    ("הוקמו",   "פורקו"),    ("פורקו",    "הוקמו"),
    ("הוקמה",   "פורקה"),    ("פורקה",    "הוקמה"),
    ("הוקם",    "פורק"),     ("פורק",     "הוקם"),
    ("הושקו",   "הופסקו"),   ("הופסקו",   "הושקו"),
    ("הושקה",   "הופסקה"),   ("הופסקה",   "הושקה"),
    ("הושק",    "הופסק"),    ("הופסק",    "הושק"),
    ("חוברה",   "נותקה"),    ("נותקה",    "חוברה"),
    ("חובר",    "נותק"),     ("נותק",     "חובר"),
]

# ── Heuristic 22: Scope/severity flip ────────────────────────────────────────
SCOPE_SEVERITY_FLIPS = [
    ("נפצעה קשות",  "נפצעה קלות"),  ("נפצעה קלות",  "נפצעה קשות"),
    ("נפצע קשות",   "נפצע קלות"),   ("נפצע קלות",   "נפצע קשות"),
    ("פגיעה קשה",   "פגיעה קלה"),   ("פגיעה קלה",   "פגיעה קשה"),
    ("באופן מלא",   "באופן חלקי"),  ("באופן חלקי",  "באופן מלא"),
    ("לחלוטין",     "חלקית"),       ("חלקית",       "לחלוטין"),
    ("מהותית",      "מינורית"),     ("מינורית",     "מהותית"),
    ("מהותי",       "מינורי"),      ("מינורי",      "מהותי"),
    ("קריטית",      "שולית"),       ("שולית",       "קריטית"),
    ("קריטי",       "שולי"),        ("שולי",        "קריטי"),
]


# ── Helper: Arabic number substitution ───────────────────────────────────────

def _sub_arabic_number(m: re.Match) -> str:
    n = int(m.group())
    # Skip plausible years: "2024" → "202" is nonsensical.
    if 1000 <= n <= 2099:
        return m.group()
    new = n * 10 if n < 1000 else n // 10
    if new == n:
        new = n + 1000
    return str(new)


def _try_arabic_number(text: str) -> Optional[tuple[str, str]]:
    if re.search(r'\b\d+\b', text):
        modified = re.sub(r'\b\d+\b', _sub_arabic_number, text, count=1)
        if modified != text:
            return modified, "number_substitution"
    return None


def _try_heb_number(text: str) -> Optional[tuple[str, str]]:
    for heb_num, replacement in HEBREW_NUMBER_MAP.items():
        pattern = r'(?<!\w)' + re.escape(heb_num) + r'(?!\w)'
        if re.search(pattern, text):
            modified = re.sub(pattern, replacement, text, count=1)
            return modified, "number_substitution_heb"
    return None


def _try_pairs(text: str, pairs: list, name: str) -> Optional[tuple[str, str]]:
    for src, tgt in pairs:
        if src in text:
            return text.replace(src, tgt, 1), name
    return None


def _try_word_pair(text: str, pairs: list, name: str) -> Optional[tuple[str, str]]:
    """Match only whole words (word boundaries)."""
    for src, tgt in pairs:
        pattern = r'(?<!\w)' + re.escape(src) + r'(?!\w)'
        if re.search(pattern, text):
            modified = re.sub(pattern, tgt, text, count=1)
            return modified, name
    return None


# ── Heuristic 13: Verb-position negation ─────────────────────────────────────
# א dropped from the prefix class: 1st-sg future (אלך) is rare in news text,
# while א-initial nouns (אופוזיציונר, אנשים) are constant false positives.
_FUT_VERB  = re.compile(r'^[יתנ][א-ת]{2,}$')
# ה suffix removed: 3rd-f-sg past (אכלה) is indistinguishable from
# definite feminine nouns (המטרה, הלשכה) without POS tagging.
_PAST_VERB = re.compile(r'^[א-ת]{2,}(?:תי|נו|תם|תן|ו)$')
_HEB_STOPWORDS = frozenset({
    # pronouns, auxiliaries, particles
    'את', 'של', 'על', 'עם', 'כי', 'אם', 'כל', 'הם', 'הן',
    'היה', 'היא', 'הוא', 'אני', 'אתה', 'זה', 'זו', 'זאת',
    'לא', 'כן', 'גם', 'רק', 'עוד', 'כבר', 'אך', 'אבל',
    # pronouns/adverbs that match the verb regexes (י/ת/נ/א-initial or ו-final)
    'אתם', 'אתן', 'אתכם', 'אתכן', 'אנחנו',
    'אותו', 'אותה', 'אותם', 'אותן', 'אותי', 'אותך', 'אותנו',
    'אתמול', 'יותר', 'יחד', 'תמיד', 'אולי', 'יאללה',
    # discourse connectives — unsafe as entity-swap targets
    'אולם', 'ואולם', 'לפיכך', 'כלומר', 'אמנם', 'ואמנם',
    'ממילא', 'לבסוף', 'ראשית', 'שנית', 'שלישית',
    'מכיוון', 'כיוון', 'משום', 'לכן', 'לכך', 'לשם',
    'אשר', 'שכן', 'ולכן', 'כפי', 'לפי', 'דרך',
})


_QUOTED_SPAN = re.compile(r'"[^"]+"|״[^״]+״|\'[^\']{2,}\'')


def _try_verb_negation(text: str) -> Optional[tuple[str, str]]:
    """Insert 'לא' before the first detected finite verb (short sentences only)."""
    tokens = text.split()
    if 'לא' in tokens or not (2 <= len(tokens) <= 12):
        return None
    # Questions don't assert anything — negating one is not a contradiction.
    if text.rstrip().endswith('?'):
        return None
    # Mark char ranges inside quoted spans (titles, names) — never negate there.
    quoted = [m.span() for m in _QUOTED_SPAN.finditer(text)]
    offset = 0
    for i, tok in enumerate(tokens):
        tok_start = text.index(tok, offset)
        offset = tok_start + len(tok)
        if any(s <= tok_start < e for s, e in quoted):
            continue
        clean = re.sub(r'[^א-ת]', '', tok)
        # Skip empty, stopwords, or tokens starting with ה (likely definite nouns)
        if not clean or clean in _HEB_STOPWORDS or clean[0] == 'ה':
            continue
        if _FUT_VERB.match(clean) or _PAST_VERB.match(clean):
            modified = tokens[:i] + ['לא'] + tokens[i:]
            return ' '.join(modified), 'verb_negation'
    return None


# ── Heuristic 14: Entity swap from premise ───────────────────────────────────

def _try_entity_swap(text: str, premise: str) -> Optional[tuple[str, str]]:
    """Replace the last content word in hypothesis with a different content word from premise."""
    hyp_tokens = text.split()
    # Stem prefixes of hypothesis words (first 3 Hebrew chars) — for morphological dedup
    hyp_stems = {
        re.sub(r'[^א-ת]', '', t)[:3]
        for t in hyp_tokens
        if len(re.sub(r'[^א-ת]', '', t)) >= 3
    }
    hyp_set = set(hyp_tokens)
    candidates = [
        w for w in premise.split()
        if w not in hyp_set
        and len(re.sub(r'[^א-ת]', '', w)) > 3
        and re.sub(r'[^א-ת]', '', w) not in _HEB_STOPWORDS
        and re.sub(r'[^א-ת]', '', w)[:3] not in hyp_stems  # block morphological variants
    ]
    if not candidates:
        return None
    # Replace the LAST content word (clause-final = main argument in Hebrew SRL)
    last_idx = None
    for i, tok in enumerate(hyp_tokens):
        clean = re.sub(r'[^א-ת]', '', tok)
        if len(clean) > 3 and clean not in _HEB_STOPWORDS:
            last_idx = i
    if last_idx is None:
        return None
    parts = hyp_tokens[:]
    parts[last_idx] = candidates[0]
    if len(parts) < 4:  # guard against fragment hypotheses
        return None
    return ' '.join(parts), 'entity_swap'


# ── Main entry point ─────────────────────────────────────────────────────────

def apply_heuristics(
    text: str,
    rng: Optional[_random.Random] = None,
    premise: Optional[str] = None,
    allowed: Optional[set] = None,
    heuristic_counts: Optional[dict] = None,
    max_fraction: float = 1.0,
) -> tuple[Optional[str], Optional[str]]:
    """
    Try all heuristics. Collect every applicable one, then randomly choose one.
    Returns (modified_text, heuristic_name) or (None, None) if nothing fires.
    entity_swap only fires when premise is provided.
    allowed: optional set of heuristic names to restrict to (None = all).
    heuristic_counts: running tally of how many times each heuristic has been
        chosen so far (updated by the caller). Combined with max_fraction, this
        caps any single heuristic at max_fraction of total contradictions,
        forcing diversity. Falls back to uncapped pool if all candidates are over
        the cap (no alternatives).
    """
    if rng is None:
        rng = _random.Random()

    candidates: list[tuple[str, str]] = []

    # 1. Arabic number substitution (skips year-range numbers)
    r = _try_arabic_number(text)
    if r: candidates.append(r)

    # 2. Hebrew number words (gender-aware)
    r = _try_heb_number(text)
    if r: candidates.append(r)

    # 4. Superlative/comparative flip (feminine + plural forms included)
    r = _try_pairs(text, SUPERLATIVE_FLIPS, "superlative_flip")
    if r: candidates.append(r)

    # 5. Temporal inversion (מלפני/מאחרי prefix forms handled)
    r = _try_pairs(text, TEMPORAL_FLIPS, "temporal_inversion")
    if r: candidates.append(r)

    # 6. Quantifier flip
    r = _try_word_pair(text, QUANTIFIER_FLIPS, "quantifier_flip")
    if r: candidates.append(r)

    # 7. Adjective antonym (feminine forms included; homographs removed)
    r = _try_word_pair(text, ADJECTIVE_ANTONYMS, "adjective_antonym")
    if r: candidates.append(r)

    # 8. Existential negation (gender/number forms included)
    r = _try_pairs(text, EXISTENTIAL_PAIRS, "existential_negation")
    if r: candidates.append(r)

    # 9. Causal reversal (NP and clause forms separated)
    r = _try_pairs(text, CAUSAL_FLIPS, "causal_reversal")
    if r: candidates.append(r)

    # 10. Modal flip
    r = _try_pairs(text, MODAL_FLIPS, "modal_flip")
    if r: candidates.append(r)

    # 11. Geographic/directional flip (ambiguous עולה/יורד removed)
    r = _try_word_pair(text, GEO_FLIPS, "geo_directional_flip")
    if r: candidates.append(r)

    # 12. Decision/approval inversion (verb + preposition paired)
    r = _try_pairs(text, DECISION_FLIPS, "decision_inversion")
    if r: candidates.append(r)

    # 13. Verb-position negation (short sentences only)
    r = _try_verb_negation(text)
    if r: candidates.append(r)


    # 15. Inclusion/exclusion flip
    r = _try_pairs(text, INCLUSION_EXCLUSION_FLIPS, "inclusion_exclusion")
    if r: candidates.append(r)

    # 16. Financial outcome flip
    r = _try_word_pair(text, FINANCIAL_FLIPS, "financial_outcome")
    if r: candidates.append(r)

    # 17. Legal status flip
    r = _try_pairs(text, LEGAL_FLIPS, "legal_status")
    if r: candidates.append(r)

    # 18. Epistemic certainty flip
    r = _try_pairs(text, EPISTEMIC_FLIPS, "epistemic_certainty")
    if r: candidates.append(r)

    # 19. Factional alliance flip
    r = _try_pairs(text, FACTIONAL_PHRASE_FLIPS, "factional_alliance")
    if r: candidates.append(r)
    if not r:
        r = _try_word_pair(text, FACTIONAL_WORD_FLIPS, "factional_alliance")
        if r: candidates.append(r)

    # 20. Pace/velocity flip
    r = _try_pairs(text, PACE_FLIPS, "pace_velocity")
    if r: candidates.append(r)

    # 21. State transition flip
    r = _try_pairs(text, STATE_TRANSITION_FLIPS, "state_transition")
    if r: candidates.append(r)

    # 22. Scope/severity flip
    r = _try_pairs(text, SCOPE_SEVERITY_FLIPS, "scope_severity")
    if r: candidates.append(r)

    if not candidates:
        return None, None

    if allowed is not None:
        candidates = [(t, h) for t, h in candidates if h in allowed]
    if not candidates:
        return None, None

    # Frequency cap: exclude over-represented heuristics while alternatives exist
    if heuristic_counts is not None and max_fraction < 1.0:
        total = sum(heuristic_counts.values())
        if total > 0:
            uncapped = [(t, h) for t, h in candidates
                        if heuristic_counts.get(h, 0) / total < max_fraction]
            if uncapped:
                candidates = uncapped

    # Inverse-frequency weighting: under-used heuristics get higher probability
    if heuristic_counts is not None:
        weights = [1.0 / (heuristic_counts.get(h, 0) + 1) for _, h in candidates]
        return rng.choices(candidates, weights=weights, k=1)[0]

    return rng.choice(candidates)


# ── Sentence-aware variant for long multi-sentence hypotheses ─────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\s*\|\s*')


def apply_heuristics_sentence_aware(
    text: str,
    rng: Optional[_random.Random] = None,
    premise: Optional[str] = None,
    allowed: Optional[set] = None,
    heuristic_counts: Optional[dict] = None,
    max_fraction: float = 1.0,
    keep_full_text: Optional[bool] = None,
) -> tuple[Optional[str], Optional[str]]:
    """
    For multi-sentence hypotheses (news summaries, article leads): try each
    sentence in order and perturb the first one that a heuristic applies to.

    Two behaviours, selected by `keep_full_text` (env: CONTRADICTION_RECIPE=v2):

    v1 (keep_full_text=False, the default) returns ONLY the modified sentence
        as the whole hypothesis, discarding the surrounding text. The intent was
        to make the contradiction salient rather than buried.

        KNOWN DEFECT: this makes hypothesis LENGTH a label cue,
        because entailment hypotheses keep the full paragraph while
        contradiction hypotheses are reduced to one sentence. Measured on HeSum:
        median contradiction hypothesis 12 tokens vs 25 for entailment; native
        Hebrew NLI shows 7-9 tokens across all three labels, i.e. no signal at
        all. A 5-feature surface classifier reaches 0.52 macro-F1 on hypothesis
        length alone. Only the two summarization converters use this function,
        and they are the two worst-transferring sources (mean delta -0.126 and
        -0.355 on held-out native NLI).

    v2 (keep_full_text=True) substitutes the modified sentence back into its
        original position and returns the FULL text, so contradiction and
        entailment hypotheses are drawn from the same length distribution. The
        contradiction is then genuinely embedded rather than length-marked --
        harder, but that is the point: the model must locate the contradiction
        instead of measuring the hypothesis.

    For single-sentence or short texts, delegates to apply_heuristics.
    allowed: optional set of heuristic names to restrict to (None = all).
    heuristic_counts / max_fraction: passed through to apply_heuristics.
    """
    if keep_full_text is None:
        keep_full_text = os.environ.get("CONTRADICTION_RECIPE", "").lower() in ("v2", "full", "full_text")

    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        return apply_heuristics(text, rng=rng, premise=premise, allowed=allowed,
                                heuristic_counts=heuristic_counts, max_fraction=max_fraction)
    for idx, sent in enumerate(sents):
        if len(sent.split()) < 4:
            continue
        modified, heuristic = apply_heuristics(sent, rng=rng, premise=premise, allowed=allowed,
                                               heuristic_counts=heuristic_counts, max_fraction=max_fraction)
        if modified:
            if keep_full_text:
                # Substitute in place; join with a period so the result reads as
                # running text regardless of what _SENT_SPLIT consumed.
                out = list(sents)
                out[idx] = modified
                return ". ".join(out), heuristic
            return modified, heuristic
    return None, None
