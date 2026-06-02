"""
03_log_odds_ratio.py
Log-odds ratio analysis following Monroe, Colaresi & Quinn (2008).
Quantifies differential keyword salience between domestic and foreign
passenger reviews using Dirichlet prior smoothing (alpha = 0.1).

Eurostar excluded (n_foreign = 74, 3.6% of Eurostar reviews).
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
ALPHA = 0.1   # Dirichlet prior smoothing parameter
MIN_SAMPLE = 50
TOP_N = 15    # Top keywords per group per cluster

CLUSTER_MAP = {
    'KTX': 1, 'Shinkansen': 1, 'THSR': 1,
    'ICE': 2, 'Frecciarossa': 2, 'Italo': 2,
    'TGV': 3, 'RENFE': 3, 'CRH': 3,
}
DOMESTIC_LANGS = {
    'KTX': ['ko'], 'Shinkansen': ['ja'], 'THSR': ['zh'],
    'ICE': ['de'], 'Frecciarossa': ['it'], 'Italo': ['it'],
    'TGV': ['fr'], 'RENFE': ['es'], 'CRH': ['zh'],
}

STOPWORDS = set([
    'the','a','an','and','but','if','or','as','of','at','by','for','with',
    'to','from','in','on','is','are','was','were','be','been','that','which',
    'has','have','had','do','did','does','not','can','will','just','so','than',
    'very','also','this','these','those','could','would','even','well','ever',
    'then','before','after','when','where','how','why','until','while',
    'i','my','me','we','our','you','your','he','his','she','her','it','its',
    'they','them','their','who','what','him',
    'train','trains','station','stations','trip','journey','travel','ride',
    'ticket','tickets','class','rail','service','seat','seats',
    'passenger','passengers','departure','arrival','time','day','way',
    'experience','first','one','get','take','make','got','went','took',
    'made','find','found','think','back','come','want','know','need',
    'good','great','nice','really','always','like','very','much','more',
    'ktx','shinkansen','thsr','crh','tgv','ice','frecciarossa','italo','renfe',
    'sncf','bahn','korail','trenitalia','bullet','ouigo',
    'seoul','busan','tokyo','kyoto','osaka','nagoya','taipei',
    'paris','london','milan','rome','berlin','munich','madrid','barcelona',
    'beijing','shanghai','florence','venice','turin',
    'japan','korea','taiwan','france','germany','italy','spain','china',
    'europe','japanese','korean','chinese','german','french','italian',
    'spanish','european','british',
])


def get_tokens(text):
    return [w for w in re.findall(r'[a-z]{3,}', str(text).lower())
            if w not in STOPWORDS and len(w) >= 4]


def compute_log_odds(c_f, c_d, vocab):
    """Monroe et al. (2008) weighted log-odds ratio with Dirichlet prior."""
    n_f = sum(c_f.values())
    n_d = sum(c_d.values())
    V = len(vocab)
    results = {}
    for w in vocab:
        lo_f = np.log((c_f[w] + ALPHA) / (n_f + V * ALPHA - (c_f[w] + ALPHA)))
        lo_d = np.log((c_d[w] + ALPHA) / (n_d + V * ALPHA - (c_d[w] + ALPHA)))
        var = 1 / (c_f[w] + ALPHA) + 1 / (c_d[w] + ALPHA)
        results[w] = (lo_f - lo_d) / np.sqrt(var)
    return results


# ── Load data ──────────────────────────────────────────────
df_reviews = pd.read_csv('../data/reviews_anonymised.csv')
df_lang = pd.read_csv('../data/language_classification.csv')
df = df_reviews.merge(df_lang[['review_id', 'train_name', 'detected_language',
                                'is_foreign']], on=['review_id', 'train_name'])

# ── Collect token counts per cluster ──────────────────────
cluster_counts = {c: {'for': Counter(), 'dom': Counter()} for c in [1, 2, 3]}

for train, c in CLUSTER_MAP.items():
    df_t = df[df['train_name'] == train]
    dom_langs = DOMESTIC_LANGS[train]
    df_d = df_t[df_t['detected_language'].isin(dom_langs)]
    df_f = df_t[~df_t['detected_language'].isin(dom_langs)]
    if len(df_d) < MIN_SAMPLE or len(df_f) < MIN_SAMPLE:
        continue
    for text in df_f['review_text'].dropna():
        cluster_counts[c]['for'].update(get_tokens(text))
    for text in df_d['review_text'].dropna():
        cluster_counts[c]['dom'].update(get_tokens(text))

# ── Compute Z-scores and extract top keywords ─────────────
all_results = []
for c in [1, 2, 3]:
    c_f = cluster_counts[c]['for']
    c_d = cluster_counts[c]['dom']
    vocab = set(c_f.keys()) | set(c_d.keys())
    zscores = compute_log_odds(c_f, c_d, vocab)

    sorted_z = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
    for_top = [(w, z) for w, z in sorted_z if z > 0][:TOP_N]
    dom_top = sorted([(w, abs(z)) for w, z in sorted_z if z < 0],
                     key=lambda x: x[1], reverse=True)[:TOP_N]

    print(f"\nCluster {c} — Foreign top {TOP_N}:")
    for w, z in for_top:
        print(f"  {w:20s}  Z = {z:.3f}")
    print(f"Cluster {c} — Domestic top {TOP_N}:")
    for w, z in dom_top:
        print(f"  {w:20s}  Z = {z:.3f}")

    for rank, (w, z) in enumerate(for_top, 1):
        all_results.append({'cluster': c, 'group': 'foreign',
                            'rank': rank, 'word': w, 'z_score': round(z, 4)})
    for rank, (w, z) in enumerate(dom_top, 1):
        all_results.append({'cluster': c, 'group': 'domestic',
                            'rank': rank, 'word': w, 'z_score': round(z, 4)})

df_out = pd.DataFrame(all_results)
df_out.to_csv('../data/log_odds_results.csv', index=False)
print("\nResults saved to data/log_odds_results.csv")
