"""
04_figures.py
Regenerates all figures used in the manuscript.

Figures produced:
  Fig 1  – Silhouette coefficient and WSS (Elbow method) by k
  Fig 2  – Dendrogram (see 01_clustering.py)
  Fig 3  – Sentiment score spider plots by cluster
  Fig 4  – Mention ratio spider plots by cluster
  Fig 5  – Domestic vs foreign sentiment scores and mention ratios
  Fig 6  – Log-odds ratio diverging bar charts by cluster
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

ATTRS = ['Tangibility', 'Reliability', 'Responsiveness',
         'Assurance', 'Empathy', 'Comfort']
ATTR_SHORT = ['Tang.', 'Reli.', 'Resp.', 'Assu.', 'Empa.', 'Comf.']
TRAIN_SYSTEMS = ['KTX', 'Shinkansen', 'THSR', 'ICE', 'Eurostar',
                 'Frecciarossa', 'Italo', 'TGV', 'RENFE', 'CRH']
CLUSTER_MAP = {'KTX': 1, 'Shinkansen': 1, 'THSR': 1,
               'ICE': 2, 'Eurostar': 2, 'Frecciarossa': 2, 'Italo': 2,
               'TGV': 3, 'RENFE': 3, 'CRH': 3}
DOMESTIC_LANGS = {
    'KTX': ['ko'], 'Shinkansen': ['ja'], 'THSR': ['zh'],
    'ICE': ['de'], 'Eurostar': ['en', 'fr', 'nl'],
    'Frecciarossa': ['it'], 'Italo': ['it'],
    'TGV': ['fr'], 'RENFE': ['es'], 'CRH': ['zh'],
}
CLUSTER_COLORS = {
    'KTX': '#1f77b4', 'Shinkansen': '#4a90d9', 'THSR': '#17becf',
    'ICE': '#d62728', 'Eurostar': '#e8603c', 'Frecciarossa': '#e377c2',
    'Italo': '#f7b6d2', 'TGV': '#2ca02c', 'RENFE': '#6abf69', 'CRH': '#bcbd22',
}
CLUSTER_NAMES = {
    1: 'Cluster 1 (KTX · Shinkansen · THSR)',
    2: 'Cluster 2 (ICE · Eurostar · Frecciarossa · Italo)',
    3: 'Cluster 3 (TGV · RENFE · CRH)',
}
DPI = 300

# ── Load ───────────────────────────────────────────────────
df = pd.read_csv('../data/sentiment_scores.csv')
df_lang = pd.read_csv('../data/language_classification.csv')

# Mean SS and MR per train
train_ss, train_mr = {}, {}
for train in TRAIN_SYSTEMS:
    df_t = df[df['train_name'] == train]
    ss, mr = {}, {}
    for attr in ATTRS:
        m = df_t[df_t[f'{attr}_sentiment'] != 'Not_mentioned'][f'{attr}_score']
        ss[attr] = float(m.mean()) if len(m) > 0 else 0.0
        mr[attr] = len(m) / len(df_t) * 100
    train_ss[train] = ss
    train_mr[train] = mr

# ── Fig 1: Silhouette + Elbow ──────────────────────────────
rows = [{'train_name': t, **train_ss[t]} for t in TRAIN_SYSTEMS]
df_c = pd.DataFrame(rows).set_index('train_name')
X = StandardScaler().fit_transform(df_c)
linked = linkage(X, method='complete')

k_vals = [2, 3, 4, 5, 6]
sil_vals, wss_vals, ch_vals = [], [], []
for k in k_vals:
    lbl = fcluster(linked, k, criterion='maxclust')
    sil_vals.append(silhouette_score(X, lbl))
    ch_vals.append(calinski_harabasz_score(X, lbl))
    wss_vals.append(sum(np.sum((X[lbl == c] - X[lbl == c].mean(0)) ** 2)
                        for c in range(1, k + 1) if len(X[lbl == c]) > 0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_vals, sil_vals, 'o-', color='#2196F3', lw=2.5, ms=9)
ax1.axvline(3, color='red', ls='--', lw=1.5, label='Selected k=3')
for k, s in zip(k_vals, sil_vals):
    ax1.annotate(f'{s:.3f}', (k, s), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')
ax1.set(xlabel='Number of Clusters (k)', ylabel='Silhouette Coefficient',
        title='Silhouette Coefficient', xticks=k_vals, ylim=(0.35, 0.75))
ax1.legend(fontsize=10)
ax1.spines[['top', 'right']].set_visible(False)

ax2.plot(k_vals, wss_vals, 's-', color='#4CAF50', lw=2.5, ms=9)
ax2.axvline(3, color='red', ls='--', lw=1.5, label='Selected k=3')
for k, w in zip(k_vals, wss_vals):
    ax2.annotate(f'{w:.2f}', (k, w), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=11, fontweight='bold')
ax2.set(xlabel='Number of Clusters (k)', ylabel='Within-Cluster Sum of Squares (WSS)',
        title='Elbow Method (WSS)', xticks=k_vals)
ax2.legend(fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)
fig.suptitle('Cluster Number Selection: Silhouette Coefficient and Elbow Method',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/fig1_k_selection.png', dpi=DPI, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")

# ── Fig 3 & 4: Spider plots ────────────────────────────────
N = len(ATTRS)
angles = [n / N * 2 * np.pi for n in range(N)] + [0]
angle_labels = ATTR_SHORT

for metric, data_dict, title_str, ylim, yticks, ytick_labels, fname in [
    ('SS', train_ss, 'Sentiment Score', (-2.3, 2.3),
     [-2, -1, 0, 1, 2], ['-2', '-1', '0', '+1', '+2'], 'fig3_spider_ss.png'),
    ('MR', train_mr, 'Mention Ratio (%)', (0, 110),
     [20, 40, 60, 80, 100], ['20%', '40%', '60%', '80%', '100%'], 'fig4_spider_mr.png'),
]:
    fig, axes = plt.subplots(1, 3, figsize=(22, 8),
                              subplot_kw=dict(projection='polar'))
    fig.suptitle(f'{title_str} by Cluster and SERVQUAL Attribute',
                 fontsize=16, fontweight='bold', y=1.02)
    for ci, c in enumerate([1, 2, 3]):
        ax = axes[ci]
        trains = [t for t in TRAIN_SYSTEMS if CLUSTER_MAP[t] == c]
        for train in trains:
            vals = [data_dict[train][a] for a in ATTRS] + [data_dict[train][ATTRS[0]]]
            ax.plot(angles, vals, 'o-', lw=2, color=CLUSTER_COLORS[train],
                    label=train, ms=5)
            ax.fill(angles, vals, alpha=0.06, color=CLUSTER_COLORS[train])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(angle_labels, fontsize=13, fontweight='bold')
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=9, color='gray')
        ax.set_title(CLUSTER_NAMES[c], fontsize=12, fontweight='bold', pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=11)
        ax.grid(color='gray', alpha=0.25)
        ax.spines['polar'].set_color('lightgray')
    plt.tight_layout()
    plt.savefig(f'../data/{fname}', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"{fname} saved.")

print("All figures saved to data/")
