# R1 package verification — 2 November 2026

The six supplied offline analysis/validation scripts were executed on the
14,764-review R1 annotations without rerunning any annotation API. The supplied
data, result tables, and figures are retained as the authors' release artifacts.
This check establishes computational behavior, not independent annotation accuracy.

Maintenance corrections:

- LICENSE: Sang-hoo Yun → Sanghoo Yoon, consistent with the author list.
- Language classification: the 316 Chinese-language THSR rows now have
  `is_foreign=False`, matching `non_domestic_language` in the primary annotations.
  Eurostar remains excluded from language-group comparisons.
- Figure output folder: `results/figures`, matching the supplied artifacts.
- Claude annotation CLI: documented model `claude-sonnet-4-6` by default,
  bundled prompt path, corrected usage examples, and an Anthropic-only model-list
  endpoint. The model-list branch was tested with fake responses, including
  pagination; no key or review text was sent to any annotation API.
- Keyword counts: `keep_default_na=False` preserves legitimate tokens such as
  `null`. Without this, pandas treats those tokens as missing and changes the
  vocabulary size and Z scores slightly. With this correction, Table 7 values
  agree with the supplied result. Equal-score ties use an alphabetical rule;
  translated-only keyword ranks 25/26 can swap without changing their Z scores.

Tables 3, the pooled Mann–Whitney Table 6, the validation metrics (S7/S12), and
bootstrap co-assignment results reproduce the supplied numerical outputs. Small
floating-point/optimizer differences occur in some ordinal-model supplementary
p-values; these do not change the unadjusted main data or supplied release tables.

## Supplementary covariate model (resolved, 4 November 2026)

The covariate model in `results/S_cov_trip_period.csv` was refitted with Newton and BFGS
optimizers after removing dummy columns that are constant within a cluster subset. Both
optimizers converged for all 18 models and reproduced the supplied table (Cluster 2 /
Tangibility: OR 2.49, adjusted p < 0.001; 12 of 18 comparisons significant in favour of
non-domestic-language reviews and one, Cluster 2 / Assurance, in favour of domestic-language
reviews). The earlier discrepancy arose from a non-converged fit that was not flagged because
warnings were suppressed. `analysis/01_reanalysis.py` and `analysis/02_supplementary.py` now
drop constant columns, use the Newton optimizer, and stop with an error if a fit does not converge.
