# Data licenses

HEntailment is derived from five native Hebrew datasets. It is released under **CC BY-SA 4.0**, the most restrictive term among its sources. If you use it, please also cite the source datasets.

| source | task | license of the source |
|---|---|---|
| HeQ (Hebrew Question Answering) | extractive QA | CC BY 4.0 |
| QASem Hebrew | semantic-role QA | CC BY 4.0 |
| HeSum | abstractive summarization | MIT |
| HebSummaries | summarization | no license stated by the authors |
| Hebrew-Paraphrase | paraphrase | CC BY-SA 4.0 |

The two translated benchmarks used in the experiments, **HebNLI** and **LCAHIM**, are not redistributed here or on the Hugging Face Hub. Obtain them from their authors and run `construction/stage1/08_hebnli_convert.py` and `04_lchaim_convert.py` to put them in the format the experiments expect. Hebrew Winograd is used the same way.
