"""
02_mann_whitney.py
Two-tailed Mann-Whitney U tests comparing domestic vs foreign passenger
sentiment scores within each cluster, with Bonferroni correction and
rank-biserial correlation effect sizes.

Eurostar excluded from language group analysis (n_foreign = 74, 3.6%).
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
ATTRS = ['Tangibility', 'Reliability', 'Responsiveness',
         'Assurance', 'Empathy', 'Comfort']

CLUSTER_MAP = {
    'KTX': 1, 'Shinkansen': 1, 'THSR': 1,
    'ICE': 2, 'Frecciarossa': 2, 'Italo': 2,          # Eurostar excluded
    'TGV': 3, 'RENFE': 3, 'CRH': 3,
}
DOMESTIC_LANGS = {
    'KTX': ['ko'], 'Shinkansen': ['ja'], 'THSR': ['zh'],
    'ICE': ['de'], 'Frecciarossa': ['it'], 'Italo': ['it'],
    'TGV': ['fr'], 'RENFE': ['es'], 'CRH': ['zh'],
}

# Bonferroni correction: 3 clusters × 6 attributes = 18 comparisons
N_TESTS = 3 * 6
ALPHA = 0.05
BONF_ALPHA = ALPHA / N_TESTS

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('../data/sentiment_scores.csv')

# ── Collect scores per cluster ─────────────────────────────
cluster_data = {c: {a: {'dom': [], 'for': []} for a in ATTRS}
                for c in [1, 2, 3]}

for train, c in CLUSTER_MAP.items():
    df_t = df[df['train_name'] == train]
    dom_langs = DOMESTIC_LANGS[train]
    df_d = df_t[df_t['language'].isin(dom_langs)]
    df_f = df_t[~df_t['language'].isin(dom_langs)]
    for attr in ATTRS:
        d = df_d[df_d[f'{attr}_sentiment'] != 'Not_mentioned'][f'{attr}_score'].dropna()
        f = df_f[df_f[f'{attr}_sentiment'] != 'Not_mentioned'][f'{attr}_score'].dropna()
        cluster_data[c][attr]['dom'].extend(d.tolist())
        cluster_data[c][attr]['for'].extend(f.tolist())

# ── Mann-Whitney U + rank-biserial ────────────────────────
def rank_biserial(f_scores, d_scores):
    n1, n2 = len(f_scores), len(d_scores)
    U, _ = mannwhitneyu(f_scores, d_scores, alternative='two-sided')
    return 1 - (2 * U) / (n1 * n2)

print(f"Bonferroni-corrected significance threshold: p < {BONF_ALPHA:.5f}")
print(f"({'alpha'} = {ALPHA} / {N_TESTS} tests)\n")

results = []
for c in [1, 2, 3]:
    n_dom = sum(len(cluster_data[c][a]['dom']) for a in ATTRS[:1])
    n_for = sum(len(cluster_data[c][a]['for']) for a in ATTRS[:1])
    print(f"Cluster {c} — domestic n≈{len(cluster_data[c]['Reliability']['dom'])}, "
          f"foreign n≈{len(cluster_data[c]['Reliability']['for'])}")
    for attr in ATTRS:
        d = np.array(cluster_data[c][attr]['dom'])
        f = np.array(cluster_data[c][attr]['for'])
        if len(d) < 5 or len(f) < 5:
            continue
        stat, p = mannwhitneyu(f, d, alternative='two-sided')
        r = rank_biserial(f, d)
        if p < BONF_ALPHA:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        results.append({
            'cluster': c, 'attribute': attr,
            'dom_mean': round(np.mean(d), 3),
            'for_mean': round(np.mean(f), 3),
            'U': round(stat, 1), 'p': round(p, 6),
            'p_bonferroni_sig': p < BONF_ALPHA,
            'rank_biserial_r': round(r, 3),
            'sig': sig
        })
        print(f"  {attr:15s}: dom={np.mean(d):.3f}, for={np.mean(f):.3f}, "
              f"p={p:.6f}, r={r:.3f}  {sig}")
    print()

df_results = pd.DataFrame(results)
df_results.to_csv('../data/mann_whitney_results.csv', index=False)
print("Results saved to data/mann_whitney_results.csv")
print(f"\nEffect size range (|r|): "
      f"{df_results['rank_biserial_r'].abs().min():.3f} – "
      f"{df_results['rank_biserial_r'].abs().max():.3f}")
print(f"Bonferroni-significant comparisons: "
      f"{df_results['p_bonferroni_sig'].sum()} / {len(df_results)}")
