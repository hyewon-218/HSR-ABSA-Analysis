"""
01_clustering.py
Hierarchical cluster analysis of 10 HSR systems
based on mean SERVQUAL sentiment scores.

Method: Complete Linkage, Euclidean distance, Z-score standardisation
Cluster validity: Silhouette coefficient, Calinski-Harabasz index,
                  Within-Cluster Sum of Squares (Elbow), Gap statistic
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
ATTRS = ['Tangibility', 'Reliability', 'Responsiveness',
         'Assurance', 'Empathy', 'Comfort']
TRAIN_SYSTEMS = ['KTX', 'Shinkansen', 'THSR', 'ICE', 'Eurostar',
                 'Frecciarossa', 'Italo', 'TGV', 'RENFE', 'CRH']

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('../data/sentiment_scores.csv')

# Compute mean sentiment score per train per attribute (mentioned only)
rows = []
for train in TRAIN_SYSTEMS:
    df_t = df[df['train_name'] == train]
    row = {'train_name': train}
    for attr in ATTRS:
        mentioned = df_t[df_t[f'{attr}_sentiment'] != 'Not_mentioned'][f'{attr}_score']
        row[attr] = float(mentioned.mean()) if len(mentioned) > 0 else 0.0
    rows.append(row)

df_mean = pd.DataFrame(rows).set_index('train_name')
print("Mean sentiment scores per system:")
print(df_mean.round(3))

# ── Standardise ────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(df_mean)
X_df = pd.DataFrame(X, index=df_mean.index, columns=ATTRS)

# ── Hierarchical clustering ────────────────────────────────
linked = linkage(X, method='complete', metric='euclidean')

# ── Cluster validity indices ───────────────────────────────
print("\nCluster validity indices:")
print(f"{'k':>4} {'Silhouette':>12} {'CH':>10} {'WSS':>10}")
for k in [2, 3, 4, 5]:
    labels = fcluster(linked, k, criterion='maxclust')
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    wss = sum(
        np.sum((X[labels == c] - X[labels == c].mean(0)) ** 2)
        for c in range(1, k + 1) if len(X[labels == c]) > 0
    )
    print(f"{k:>4} {sil:>12.4f} {ch:>10.4f} {wss:>10.4f}")

# ── Gap statistic ──────────────────────────────────────────
def gap_statistic(X, linked, k_range, B=100, seed=42):
    rng = np.random.default_rng(seed)
    results = []
    for k in k_range:
        labels = fcluster(linked, k, criterion='maxclust')
        wss_obs = sum(
            np.sum((X[labels == c] - X[labels == c].mean(0)) ** 2)
            for c in range(1, k + 1) if len(X[labels == c]) > 0
        )
        log_wss_refs = []
        for _ in range(B):
            X_ref = rng.uniform(X.min(0), X.max(0), X.shape)
            lnk_ref = linkage(X_ref, method='complete')
            lbl_ref = fcluster(lnk_ref, k, criterion='maxclust')
            wss_ref = sum(
                np.sum((X_ref[lbl_ref == c] - X_ref[lbl_ref == c].mean(0)) ** 2)
                for c in range(1, k + 1) if len(X_ref[lbl_ref == c]) > 0
            )
            log_wss_refs.append(np.log(wss_ref + 1e-10))
        gap = np.mean(log_wss_refs) - np.log(wss_obs + 1e-10)
        se = np.std(log_wss_refs)
        results.append({'k': k, 'gap': gap, 'se': se})
    return pd.DataFrame(results)

print("\nGap statistic (B=100):")
gap_df = gap_statistic(X, linked, [2, 3, 4, 5])
print(gap_df.round(4).to_string(index=False))

# ── Final solution: k = 3 ──────────────────────────────────
labels_k3 = fcluster(linked, 3, criterion='maxclust')
cluster_assignments = dict(zip(df_mean.index, labels_k3))
print("\nCluster assignments (k=3):")
for train, c in cluster_assignments.items():
    print(f"  {train}: Cluster {c}")

# ── Plot dendrogram ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(linked, labels=df_mean.index, ax=ax,
           color_threshold=3.5, above_threshold_color='gray')
ax.axhline(y=3.5, color='red', linestyle='--', linewidth=1.5,
           label='Cut-off (k=3)')
ax.set_title('Dendrogram — Hierarchical Cluster Analysis of 10 HSR Systems\n'
             '(Complete Linkage, Euclidean distance, Z-score standardised)',
             fontsize=12)
ax.set_ylabel('Distance')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('../data/dendrogram.png', dpi=300, bbox_inches='tight')
print("\nDendrogram saved to data/dendrogram.png")
