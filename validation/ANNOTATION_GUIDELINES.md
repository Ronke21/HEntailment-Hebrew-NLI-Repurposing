# HEntailment Validation — Annotation Guidelines

315 rows. For each, read the premise and hypothesis and decide the relationship between them. Fill `annotator_label` with **exactly one** of these four values (nothing else — no explanations):

| label | meaning |
|---|---|
| `היסק` | Hypothesis **must be true**, given the premise. |
| `סתירה` | Hypothesis **must be false**, given the premise — both cannot be true. |
| `ניטרלי` | Hypothesis **may be true or false** — the premise doesn't determine it. |
| `לא-תקין` | The hypothesis isn't coherent Hebrew / isn't a real sentence — meaning can't be judged. |

## Rules

1. Judge using **only** the premise text given. No outside knowledge, no assumptions beyond what's stated.
2. A paraphrase of the premise (same content, different words) = `היסק`, even if the wording is very different.
3. A plausible but unstated detail (could be true, but isn't required by the premise) = `ניטרלי`. "Sounds likely" is not enough for `היסק`.
4. A hypothesis that reverses or negates a fact the premise states explicitly = `סתירה`, even if only one word changed.
5. If the logic is easy but the Hebrew itself is broken/garbled, that's `לא-תקין`, not a guess at the logic. Don't force a label onto text you can't parse.
6. One label per row, every row, in order. Don't skip rows.

## Pronouns and coreference

**When the hypothesis contains a pronoun, assume it refers to an entity mentioned in the premise.** This is the standard NLI convention (inherited from SNLI/MNLI): both sentences describe the same situation, so pronouns are understood as referring back.

**Example:**
- Premise: "Dana took the book."
- Hypothesis: "She took the book."
- **Label: `היסק`** — "she" refers to Dana, the predicate and object are identical, so the hypothesis is entailed.

**When identity is genuinely ambiguous** (e.g., multiple people in the premise, or a pronoun with no clear antecedent), treat it as `ניטרלי` — the truth of the hypothesis is underdetermined without knowing which referent the pronoun picks out. Do not guess at the referent.

## Examples (illustrative only — not from your file)

| premise | hypothesis | label |
|---|---|---|
| הכלב רץ בגינה. | בעל חיים היה בגינה. | `היסק` |
| החנות פתוחה בימי ראשון. | החנות סגורה בימי ראשון. | `סתירה` |
| דנה קנתה ספר אתמול. | דנה קראה את הספר. | `ניטרלי` |
| *(any garbled/non-Hebrew hypothesis)* | | `לא-תקין` |

## Independence

- Work alone. Don't discuss items or compare answers with the other annotator until **both** files are submitted.
- Don't open any other file in this folder — some contain the correct answers and would invalidate the study.

## When done

Save your file in place. Confirm every row has exactly one of the four values above, then send it back.
