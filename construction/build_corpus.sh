#!/usr/bin/env bash
# Build HEntailment from the source datasets.
#   bash construction/build_corpus.sh            # all five sources
#   SOURCES="heq" bash construction/build_corpus.sh
# Needs: the source datasets under data/original/ (see DATA_LICENSE.md);
# OPENROUTER_API_KEY for stage 1's LLM realization of HeQ/QASem entailments;
# one GPU with vLLM for stage 2d (gemma-4-31B-it).
# Stage 2a/2c and 2d accept --shard I/N, so they can be spread over several jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export HENT_ROOT=$PWD
SOURCES=${SOURCES:-"heq qasem hesum hebsummaries heb_paraphrase"}
ORIG=data/original
S1=data/stage1
# similarity-selected neutrals and full-text hypotheses (the recipe of the released corpus)
export NEUTRAL_RECIPE=v2 CONTRADICTION_RECIPE=v2

# ---- stage 1: source datasets -> entailment and neutral pairs
for s in $SOURCES; do
  case $s in
    heq)            python construction/stage1/01_heq_convert.py --data_dir "$ORIG/Hebrew-Question-Answering-Dataset/data/data v1.1" --output_dir $S1/heq ;;
    hesum)          python construction/stage1/02_hesum_convert.py --data_dir $ORIG/HeSum/data --output_dir $S1/hesum ;;
    heb_paraphrase) python construction/stage1/03_heb_paraphrase_convert.py --data_file $ORIG/Hebrew-Paraphrase-Dataset/Dataset.json --output_dir $S1/heb_paraphrase ;;
    hebsummaries)   python construction/stage1/05_hebsummaries_convert.py --data_dir $ORIG/HebSummaries --output_dir $S1/hebsummaries ;;
    qasem)          python construction/stage1/06_qasem_convert.py --data_dir $ORIG/QASem_Hebrew --output_dir $S1/qasem --train_sample 20000 ;;
  esac
done

# ---- stage 2: rule-based contradictions, judged
for s in $SOURCES; do
  mkdir -p data/corpus/$s
  cp construction/resources/$s/exclude_rules.txt data/corpus/$s/
  python construction/stage2/gen_candidates_full.py --src $s          # 2a
  python construction/stage2/gen_placeholder_rows.py --src $s         # 2b
  python construction/stage2/add_ungrounded.py --src $s               # 2c
  python construction/stage2/merge_ungrounded.py --src $s
  python construction/stage2/judge_full.py --src $s                   # 2d (GPU)
  python construction/stage2/select_full.py --src $s                  # 2e
  python construction/stage2/run_gate.py --src $s --dir data/corpus/$s   # 2f
done
echo "corpus written to data/corpus/<source>/{train,dev,test}.jsonl"
