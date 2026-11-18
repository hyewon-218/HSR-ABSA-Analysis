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
| `data/claude_sonnet46_annotations.csv` | Label, score (−2 to +2), and confidence for six attributes (verbatim evidence excerpts removed; see below), 14,764 reviews; `non_domestic_language` is empty for Eurostar |
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
| `05_empathy_price_terms.py` | Section 3.1 (share of empathy excerpts with a price-related term); requires the evidence excerpts, which are not public, so the output is provided in `results/` |
| `validation/eval_validation.py` | Section 3.1; Tables S7, S12 |

Re-running the annotation (`annotation/annotate_claude_full.py`) requires the review texts and an API key; even with a fixed model version and temperature 0, a repeated run reproduced aspect detection with κ = 0.81.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0); see `LICENSE`.

## Version note (revision R2)

Added for the second revision:

- `analysis/06_quality_screen.py` — data-quality screening (language re-detection with langdetect, p ≥ 0.95; platform residue; duplicate texts). Requires the original review texts, which are not redistributed because of the platform's terms of use; the resulting flags are provided.
- `results/R2_quality/quality_flags.csv` — per-review flags (recorded and detected language, detection probability, language mismatch, platform residue, duplicate).
- `results/R2_quality/excluded_review_ids.csv` — the 11 flagged reviews.
- `results/R2_quality/filtered_*` — main results re-computed without the flagged reviews (`python analysis/01_reanalysis.py <annotations without flagged reviews> <outdir>`); Table S13.
- `analysis/07_star_attribute_analysis.py` — relation between attribute-level sentiment and the star rating (discordance, cumulative logit attribute weights with system fixed effects, language-group comparison, Shapley decomposition of R²); Section 3.4 and Table 8. Run from the repository root: `python analysis/07_star_attribute_analysis.py`.
- `results/R2_star/star_attr.json` — output of the star-rating analysis.

No human ground truth was constructed; the validation files in `validation/` are model-based reference annotations (see the manuscript, Sections 2.2 and 4.5).

## Removal of verbatim evidence excerpts (November 2026)

At the request of the journal regarding the use of TripAdvisor content, the verbatim evidence excerpts (`*_evidence` columns, the `te`/`me` columns of `validation/annotator_pairs.csv` and `validation/reference.csv`, and the corresponding fields in `validation/adjudication.json`) were removed from all public files. Labels, scores, confidence values, metadata, aggregated word counts, and code are unchanged. The excerpts and review texts are available to editors and reviewers for verification on request to the corresponding author. Scripts that read the excerpts (e.g., `analysis/05_empathy_price_terms.py`) cannot be re-run from the public files; their outputs remain in `results/`.
