# Data licenses

HEntailment is derived from five native Hebrew datasets. The annotations and processing we add (labels, contradiction edits, neutral pairing, splits) are released under **CC BY-SA 4.0**, the most restrictive license among the sources that state one. Text from a source without a stated license (HebSummaries) stays under its rights holders' terms; we do not grant rights to it. If you use the corpus, please also cite the source datasets.

| source | task | license of the source |
|---|---|---|
| HeQ (Hebrew Question Answering) | extractive QA | CC BY 4.0 |
| QASem Hebrew | semantic-role QA | CC BY 4.0 |
| HeSum | abstractive summarization | MIT |
| HebSummaries | summarization | no license stated by the authors |
| Hebrew-Paraphrase | paraphrase | CC BY-SA 4.0 |

The two translated benchmarks used in the experiments, **HebNLI** and **LCHAIM**, are not redistributed here or on the Hugging Face Hub. Obtain them from their authors and run `construction/stage1/08_hebnli_convert.py` and `04_lchaim_convert.py` to put them in the format the experiments expect. Hebrew Winograd is used the same way.
