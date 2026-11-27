# HEntailment: native Hebrew NLI by repurposing

Code, results and instructions for **HEntailment**, a native Hebrew natural language inference (NLI) benchmark of **173,460 premise–hypothesis pairs**. It is built by converting five native Hebrew datasets (question answering, summarization, paraphrase, semantic-role QA) into NLI, without translation.

> Ron Keinan and Reut Tsarfaty. *Lost in Translation, Found in Other Benchmarks: Cross-Task Transfer for Natural Language Inference.* EACL 2027.

| | |
|---|---|
| Dataset | [huggingface.co/datasets/ronke21/HEntailment](https://huggingface.co/datasets/ronke21/HEntailment) |
| Models | [huggingface.co/ronke21](https://huggingface.co/ronke21): fine-tuned encoders and LoRA adapters (see [Models](#models)) |
| Paper | arXiv link to follow |

## Load the data

```python
from datasets import load_dataset
ds = load_dataset("ronke21/HEntailment", "all")      # or one source: "heq", "qasem", "hesum", "hebsummaries", "heb_paraphrase"
print(ds["test"][0])   # {"id", "premise", "hypothesis", "label", "source", ...}
```

Labels are `entailment`, `contradiction` and `neutral`. A contradiction row also records the rule(s) that produced it (`rules`), the entailment it was derived from (`source_entailment`), the number of edited tokens, and whether the replaced token occurs in the premise (`grounded`).

| source | train | dev | test | entailment | contradiction | neutral |
|---|---:|---:|---:|---:|---:|---:|
| HeQ | 47,505 | 2,580 | 2,639 | 21,784 | 9,156 | 21,784 |
| QASem | 42,852 | 781 | 1,722 | 21,157 | 3,128 | 21,070 |
| HeSum | 19,072 | 2,382 | 2,437 | 9,997 | 3,897 | 9,997 |
| HebSummaries | 10,111 | 2,091 | 2,102 | 5,024 | 4,328 | 4,952 |
| Hebrew-Paraphrase | 29,655 | 3,818 | 3,713 | 17,382 | 9,958 | 9,846 |
| **total** | 149,195 | 11,652 | 12,613 | 75,344 | 30,467 | 67,649 |

The test splits keep the label distribution the construction produces. Models are trained on a class-balanced sample of 1,000 rows per label per source (`experiments/build_data_root.py`).

The dataset also has a **`verified` test config**: the test items whose label an independent judge (Qwen3.6-27B, a model family not used to build the corpus) confirms. It is a cleaner but easier evaluation set, biased toward items an LLM finds clear. Use it alongside the full test split, not instead of it. Qwen3.6-27B scores on `verified` are not comparable with other models' scores on it, because the same model chose the items.

## How the corpus is built

Each source annotation is mapped to NLI pairs whose label follows from the original task.

1. **Entailment.** The gold annotation becomes the hypothesis: a question and its answer are rewritten as a declarative sentence (an LLM does the surface realization only), a summary is entailed by its article, a paraphrase by its original.
2. **Neutral.** For each premise, the most similar hypothesis written for a *different* premise (NeoDictaBERT embedding similarity, below a cosine ceiling of 0.7).
3. **Contradiction.** Rule-based edits of the entailment hypothesis (swap a person, organization or place for another entity of the same type, change a number to one of the same magnitude, antonym, superlative, quantifier, ...). gemma-4-31B-it then **filters** the candidates. A candidate ships only if the judge finds the edited sentence grammatical *and* calls the pair a contradiction. Rules whose output is recoverable from the hypothesis alone are excluded (`construction/resources/<src>/exclude_rules.txt`). Grounded edits are preferred.

| stage | script | hardware |
|---|---|---|
| 1. source datasets → entailment and neutral pairs | `construction/stage1/0*_convert.py` | CPU, plus an LLM API for HeQ/QASem realization |
| 2a. contradiction candidates | `construction/stage2/gen_candidates_full.py` | CPU, sharded |
| 2b. candidates for QASem placeholder rows | `construction/stage2/gen_placeholder_rows.py` | CPU |
| 2c. ungrounded candidates | `construction/stage2/add_ungrounded.py`, `merge_ungrounded.py` | CPU |
| 2d. judge (grammar + label) | `construction/stage2/judge_full.py` | 1 GPU, vLLM |
| 2e. select one contradiction per entailment, write splits | `construction/stage2/select_full.py` | CPU |
| 2f. per-rule artifact gate | `construction/stage2/run_gate.py` | CPU |

`construction/build_corpus.sh` runs the stages in order. The source datasets are not redistributed here; download them from their authors (links in `DATA_LICENSE.md`) into `data/original/`.

Quality checks on the finished corpus are in `quality/`:
- `second_judge.py`: independent-judge validity of a contradiction sample;
- `verify_test.py`: builds the `verified` subset;
- `hyponly_*.py`, `corpus_probes.py`: hypothesis-only and premise-only probes, and label mix;
- `run_aflite.py`, `apply_aflite.py`: AFLite filtering and re-scoring.

The human-validation protocol is in `validation/` (sampling, guidelines, scoring). The human annotations themselves are not released.

## Experiments

`experiments/src/run_experiments.py` fine-tunes and evaluates one model on one training condition. Experiment configs:

- `experiments/grid`: 11 models × 8 training conditions × 3 seeds, each scored on 8 test sets;
- `experiments/s3`: the low-resource HebNLI curve;
- `experiments/ablation`: the neutral-selection ablation.

```bash
pip install -r requirements.txt
python experiments/build_data_root.py          # data/corpus -> data/experiments (needs HebNLI and LCHAIM, see DATA_LICENSE.md)
python experiments/src/run_experiments.py \
    --experiment-root experiments/grid \
    --models FacebookAI/xlm-roberta-large --train-datasets hent_all --seeds 42
```

Hyperparameters are in `experiments/<exp>/src/config.py`:
- **Encoders:** full fine-tuning with a 3-way head, lr 2e-5, batch 32, up to 20 epochs, best checkpoint by dev macro-F1 (patience 5).
- **Decoders:** LoRA causal-LM SFT (r 16, all linear layers), lr 1e-4, up to 3 epochs, paged AdamW 8-bit. The targets are the three Hebrew label words; evaluation is greedy generation followed by label normalisation.

On V100 GPUs set `FORCE_FP32=1`.

**Reproduce the paper's tables.** Every run's `metrics.json` is in `experiments/{grid,s3,ablation}/results/` (per-run predictions are in the dataset repository under `predictions/`). The tables are rebuilt from these files on CPU:

```bash
python experiments/paper_tables.py         # transfer, in-domain, zero-shot, full matrices
python experiments/transfer_report.py      # both transfer directions, model-level tests
python experiments/s3_ablation_report.py   # low-resource curve and neutral ablation
python experiments/rescore_balanced.py --root experiments/grid   # needs predictions/
```

## Models

All models are trained with seed 42, on the pooled corpus (`hent_all`) or on HebNLI-15k:
- **Encoders** (pooled HEntailment): NeoDictaBERT, AlephBERT, XLM-R-large, XLM-R-XL, mE5-large.
- **Decoder LoRA adapters** (pooled HEntailment and HebNLI-15k): DictaLM-3.0-1.7B-Instruct, DictaLM-3.0-24B-Base, Hebatron, gemma-4-12B-it, gemma-4-31B-it, Qwen3.6-27B.
  - The adapters load on top of the original base models.
  - Qwen3.6-27B's adapter must be evaluated with Hugging Face PEFT; vLLM does not apply it.

Each model card lists its scores (mean ± SD over the three seeds, from the paper).

## Licenses

- **Code:** Apache-2.0 (`LICENSE`).
- **Data:** our annotations and processing are CC BY-SA 4.0, the most restrictive license among the sources that state one. Text from a source without a stated license (HebSummaries) stays under its rights holders' terms. Per-source terms are in `DATA_LICENSE.md`.

## Citation

```bibtex
@inproceedings{keinan-tsarfaty-2027-hentailment,
  title     = {Lost in Translation, Found in Other Benchmarks: Cross-Task Transfer for Natural Language Inference},
  author    = {Keinan, Ron and Tsarfaty, Reut},
  booktitle = {Proceedings of the Conference of the European Chapter of the Association for Computational Linguistics (EACL 2027)},
  year      = {2027}
}
```
