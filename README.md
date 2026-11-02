# HSR-ABSA-Analysis

Data, annotation prompt, annotation log, validation data, and analysis code for:

**Benchmarking High-Speed Rail Service Quality with Large Language Model-Based Aspect-Based Sentiment Analysis of Multilingual Online Reviews** (*Applied Sciences*, under review)

Hyewon Lim, Wonjae Lee, Byungho Chung, Junghyun Lim, Sanghoo Yoon

## Version note (revision R1)

The annotations released with the original submission were replaced. A data-processing check showed that, for 35% of the reviews, the text passed to the annotation model did not correspond to the crawled review. All 14,764 reviews were therefore re-annotated from the original texts with the fixed prompt (`prompt/SAP_prompt_v1.txt`, fixed on 5 April 2026) through a version-fixed API:

- Primary annotation: Claude Sonnet 4.6 (`claude-sonnet-4-6`), Anthropic Messages API, temperature 0, batches of 30 reviews from the same system, 29–30 October 2026 (UTC). The model identifier and response identifier of every batch are in `annotation/annotation_log.csv`.
- Comparison model: Gemini 3 Flash (`gemini-3-flash-preview`), Gemini API, temperature 0, validation sample only.

## What is (and is not) included

Review texts collected from TripAdvisor (10–29 December 2025) are **not redistributed**, in accordance with the TripAdvisor terms of use. They are available from the corresponding author for verification on reasonable request. Everything else needed to reproduce the analyses is included.

| Path | Content |
|---|---|
| `data/claude_sonnet46_annotations.csv` | Label, score (−2 to +2), short verbatim evidence excerpt, and confidence for six attributes, 14,764 reviews; `non_domestic_language` is empty for Eurostar |
| `data/language_classification.csv` | Detected review language (`langdetect`) and language group |
| `data/review_metadata.csv` | Star rating, date of writing, month of visit, and trip type shown on the review page |
| `data/chinese_script.csv` | Script (simplified/traditional) of the original Chinese reviews, for the sensitivity analysis |
| `data/word_counts_by_system_group.csv` | Token counts (English translations; four or more letters) by system, language group, and English/non-English source, from which the keyword analysis is reproduced |
| `prompt/SAP_prompt_v1.txt` | Full annotation prompt (Appendix A of the paper) |
| `annotation/` | Annotation scripts (API keys are read from environment variables) and the batch log |
| `validation/` | Validation sample identifiers, the two independent reference annotations, adjudication, reference, test–retest annotation, Gemini annotations, and the evaluation script |
| `results/` | All result tables (Tables 3–7, S1–S12), `results.json`, and figures |

### Validation sample

`validation/validation_sample_ids.csv` lists 400 reviews (50 in each of eight languages). The reference annotations in `validation/reference_annotator_*.csv` were produced by Claude Fable 5.1 (`claude-fable-5-1`) under two expert roles (transport studies; services management), blind to the primary annotation and to each other; disagreements were adjudicated by a third instance (`validation/adjudication.json`). These are language-model annotations, not human annotations. Blank sheets for human annotation of the same reviews are available from the corresponding author because they contain the review texts.

## Reproducing the analyses

Verification notes and one supplementary-analysis discrepancy are recorded in
[`REPRODUCIBILITY_NOTES.md`](REPRODUCIBILITY_NOTES.md). Supplied result files are
retained; computational verification does not independently validate the reviews
or establish annotation accuracy. On Windows, run with Python UTF-8 mode enabled
(`python -X utf8 ...` or `PYTHONUTF8=1`).


```bash
pip install -r requirements.txt
python analysis/01_reanalysis.py data/claude_sonnet46_annotations.csv results   # Tables 3-6, S1-S4, S9, S11; clustering, bootstrap, PCA
python analysis/02_supplementary.py data/claude_sonnet46_annotations.csv results # Tables S8, S10, S11 (script), speed CI
python analysis/03_keywords.py results                                            # Table 7, S5, Figure 6 input
python analysis/04_figures.py data/claude_sonnet46_annotations.csv results      # Figures 1-6, S1, S2
python validation/eval_validation.py                                              # Tables S7, S12
```

| Script | Paper |
|---|---|
| `01_reanalysis.py` | Sections 3.1–3.3.1; Tables 3–6, S1–S4, S9, S11 (covariates, design effects) |
| `02_supplementary.py` | Tables S8 (error simulation), S10 (k = 2, 4), S11 (Chinese script) |
| `03_keywords.py` | Section 3.3.2; Table 7, Table S5, Figure 5–6 input |
| `04_figures.py` | Figures 1–6, S1, S2 |
| `05_empathy_price_terms.py` | Section 3.1 (share of empathy excerpts with a price-related term) |
| `validation/eval_validation.py` | Section 3.1; Tables S7, S12 |

Re-running the annotation (`annotation/annotate_claude_full.py`) requires the review texts and an API key; even with a fixed model version and temperature 0, a repeated run reproduced aspect detection with κ = 0.81.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0); see `LICENSE`.

## Repository maintenance (2 November 2026)

This revision replaces the superseded Gemini-era dataset and analysis scripts.
The supplied R1 data, validation files, and figures are retained. The local
handoff's Gemini comparison files are included; no new API annotation was run.

Minor implementation corrections: LICENSE uses Sanghoo Yoon; the language table's
316 Chinese-language THSR rows are marked domestic, matching the primary
annotations (Eurostar remains excluded from language-group comparisons); figure
exports use results/figures. The Claude annotation CLI defaults to the documented
claude-sonnet-4-6, reads the bundled prompt by path, and lists models using only
the Anthropic endpoint. This does not change any stored annotation or analysis.

Full review texts/translations are removed from the current revision. Short
annotation evidence excerpts provided in the R1 package remain. Removing a file
in this commit does not purge it from earlier Git history. History cleanup, if
required, must be coordinated separately; this update does not rewrite history.

Keyword CSV parsing preserves literal words such as `null` instead of treating
them as missing values. Equal-score keyword ties are sorted deterministically.
