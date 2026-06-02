# HSR-ABSA-Analysis

**Comparative Service Quality Assessment of Global High-Speed Rail Systems Using LLM-Based Aspect-Based Sentiment Analysis of Online Reviews**

Hyewon Lim, Sang-hoo Yun (Department of Statistics, Chonnam National University)

---

## Overview

This repository contains all data, analysis code, and the structured annotation prompt for the paper submitted to *PLOS ONE*.

We apply **LLM-based Aspect-Based Sentiment Analysis (ABSA)** to 14,764 multilingual TripAdvisor reviews from 10 major high-speed rail (HSR) systems across four continents. Sentiment scores are extracted for six SERVQUAL dimensions, then used in hierarchical cluster analysis and domestic–foreign passenger perception comparisons.

**Systems analysed:** KTX (South Korea), Shinkansen (Japan), THSR (Taiwan), ICE (Germany), Eurostar (UK/France), Frecciarossa (Italy), Italo (Italy), TGV (France), RENFE AVE (Spain), CRH (China)

---

## Repository Structure

```
HSR-ABSA-Analysis/
│
├── README.md
├── requirements.txt
├── LICENSE                            ← CC BY 4.0
│
├── data/
│   ├── reviews_anonymised.csv         ← Review texts (reviewer names removed)
│   ├── sentiment_scores.csv           ← LLM annotation results (scores + evidence)
│   ├── language_classification.csv    ← Language detection results (domestic/foreign)
│   ├── mann_whitney_results.csv       ← Mann-Whitney U test results (generated)
│   └── log_odds_results.csv           ← Log-odds ratio results (generated)
│
├── prompt/
│   └── SAP_prompt_v1.txt              ← Full structured annotation prompt
│
└── analysis/
    ├── 01_clustering.py               ← Hierarchical cluster analysis (k=3)
    ├── 02_mann_whitney.py             ← Mann-Whitney U + Bonferroni + rank-biserial
    ├── 03_log_odds_ratio.py           ← Monroe et al. (2008) log-odds ratio
    └── 04_figures.py                  ← All manuscript figures
```

---

## Data Collection

- **Platform:** TripAdvisor (publicly accessible pages)
- **Method:** Web crawling for non-commercial academic research
- **Total reviews:** 14,764
- **Languages:** English, Korean, Japanese, Chinese, German, French, Italian, Spanish
- **Domestic/foreign classification:** Review language detected via `langdetect` library

**System-level review counts:**

| System | Country | Route | Reviews |
|--------|---------|-------|---------|
| KTX | South Korea | Seoul–Busan | 1,178 |
| Shinkansen | Japan | Tokyo–Osaka | 1,626 |
| THSR | Taiwan | Taipei–Zuoying | 1,150 |
| ICE | Germany | Berlin–Munich | 2,093 |
| Eurostar | UK/France | London–Paris | 2,054 |
| Frecciarossa | Italy | Milan–Rome | 611 |
| Italo | Italy | Milan–Naples | 1,034 |
| TGV | France | Paris–Lyon | 2,300 |
| RENFE AVE | Spain | Madrid–Barcelona | 718 |
| CRH | China | Beijing–Shanghai | 2,000 |

---

## Methods Summary

1. **LLM-based ABSA** — Gemini 3 Flash (web interface, batches of 30–50 reviews) extracts per-attribute sentiment scores (−2 to +2) across six SERVQUAL dimensions. See `prompt/SAP_prompt_v1.txt` for the full annotation protocol.

2. **Hierarchical cluster analysis** — Complete Linkage, Euclidean distance, Z-score standardised sentiment scores. Optimal k=3 selected via silhouette coefficient, Calinski-Harabasz index, WSS Elbow method, and Gap statistic.

3. **Mann-Whitney U test** — Two-tailed, Bonferroni-corrected (α = 0.05 / 18 = 0.00278), with rank-biserial correlation effect sizes. Compares domestic vs foreign passenger sentiment scores within each cluster.

4. **Log-odds ratio analysis** — Monroe, Colaresi & Quinn (2008) weighted log-odds ratio with Dirichlet prior smoothing (α = 0.1). Quantifies differential keyword salience between domestic and foreign reviewers.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run analyses in order
cd analysis
python 01_clustering.py      # Cluster analysis + validity indices
python 02_mann_whitney.py    # Statistical tests
python 03_log_odds_ratio.py  # Keyword salience analysis
python 04_figures.py         # Regenerate all figures
```

---

## Citation

> Lim H, Yun S. Comparative service quality assessment of global high-speed rail systems using LLM-based aspect-based sentiment analysis of online reviews. *PLOS ONE*. 2026. doi:[to be added after publication]

---

## License

This repository is licensed under [CC BY 4.0](LICENSE).
Data are derived from publicly available TripAdvisor pages collected for non-commercial academic research.
