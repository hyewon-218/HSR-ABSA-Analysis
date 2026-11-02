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

## Supplementary covariate model requires author review

For `results/S_cov_trip_period.csv`, this machine's rerun does not reproduce all
supplied p-values. The clearest difference is Cluster 2 / Tangibility: the supplied
adjusted p-value is approximately 3.85e-10, whereas the rerun reports 1.0 despite
a similar odds ratio. The script suppresses fitting warnings, so successful exit
alone does not validate this fit. Solver/version sensitivity or model numerical
stability should be investigated by the authors; the cause was not established
by this repository-maintenance update. The supplied table is preserved rather
than silently replaced with the rerun output.

## Data release boundary

Full review texts/translations and the superseded annotation CSV are removed
from the current tree. Short evidence excerpts explicitly supplied in R1 remain.
This normal commit does not purge old review text from earlier commits. Any
history rewrite must be coordinated with the repository owner and collaborators.
No history rewrite or optional DOI/Zenodo release was performed.

The model-list endpoint follows the [Anthropic API reference](https://platform.claude.com/docs/en/api/models/list).
