"""Figures for the reanalysis. Usage: python figures_new.py ANNOTATION.csv OUTDIR"""
import sys, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
ANN, OUT = sys.argv[1], sys.argv[2]; FIG = f'{OUT}/figures'; os.makedirs(FIG, exist_ok=True)
R = json.load(open(f'{OUT}/results.json')); CLU = R['clusters']
A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']; AS = ['Tang.', 'Reli.', 'Resp.', 'Assu.', 'Empa.', 'Comf.']
T = ['KTX', 'Shinkansen', 'THSR', 'ICE', 'Eurostar', 'Frecciarossa', 'Italo', 'TGV', 'RENFE', 'CRH']
NM = {t: ('RENFE AVE' if t == 'RENFE' else t) for t in T}
CC = {1: '#2b6cb0', 2: '#c0392b', 3: '#2e8b57'}  # 1 high, 2 low, 3 intermediate
PAL = ['#1f4e9c', '#3b7dd8', '#86b6ef', '#8e1b1b', '#d9472b', '#e8838f', '#f2b8a0', '#1b6e35', '#5cb85c', '#b5c935']
SS = pd.read_csv(f'{OUT}/table3_SS.csv', index_col=0).loc[T]; MR = pd.read_csv(f'{OUT}/table3_MR.csv', index_col=0).loc[T]
plt.rcParams.update({'font.size': 11})
X = StandardScaler().fit_transform(SS.values); L = linkage(X, 'complete')
h = sorted(L[:, 2]); cut = (h[-3] + h[-2]) / 2
# Figure 1
fig, ax = plt.subplots(figsize=(10, 5.5)); set_link_color_palette(['#777777'])
dendrogram(L, labels=[NM[t] for t in T], ax=ax, color_threshold=0, above_threshold_color='#555555')
ax.axhline(cut, ls='--', color='k', lw=1, label='Cut-off (k = 3)')
for lbl in ax.get_xticklabels():
    t = [k for k, v in NM.items() if v == lbl.get_text()][0]; lbl.set_color(CC[CLU[t]]); lbl.set_fontweight('bold')
ax.set_ylabel('Euclidean distance'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
plt.xticks(rotation=40, ha='right'); plt.tight_layout(); plt.savefig(f'{FIG}/Figure1.png', dpi=600); plt.close()
R['cut_height'] = round(float(cut), 2)
# Figures 2, 3
ang = [n / 6 * 2 * np.pi for n in range(6)] + [0]
QP = ['#1b4f9c', '#d9711c', '#2a9d8f', '#7b4fa0']  # distinct line colours within each panel
COL = {}
for c in [1, 2, 3]:
    for j, t in enumerate([t for t in T if CLU[t] == c]): COL[t] = QP[j]
TITLE = {1: 'Cluster 1 (high)', 2: 'Cluster 2 (low)', 3: 'Cluster 3 (intermediate)'}
for data, yl, yt, ytl, fn in [(SS, (-2.2, 2.2), [-2, -1, 0, 1, 2], ['−2', '−1', '0', '+1', '+2'], 'Figure2.png'), (MR, (0, 105), [20, 40, 60, 80, 100], ['20', '40', '60', '80', '100'], 'Figure3.png')]:
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), subplot_kw=dict(projection='polar'))
    for i, c in enumerate([1, 2, 3]):
        a = axs[i]
        for t in [t for t in T if CLU[t] == c]:
            v = list(data.loc[t]) + [data.loc[t].iloc[0]]
            a.plot(ang, v, 'o-', lw=2, ms=4, color=COL[t], label=NM[t]); a.fill(ang, v, alpha=0.05, color=COL[t])
        a.set_xticks(ang[:-1]); a.set_xticklabels(AS, fontsize=11); a.set_ylim(yl); a.set_yticks(yt); a.set_yticklabels(ytl, fontsize=8, color='gray')
        a.set_title(TITLE[c], fontsize=12, fontweight='bold', color=CC[c], pad=14)
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    plt.tight_layout(); plt.savefig(f'{FIG}/{fn}', dpi=600); plt.close()
# Figure S1
ks = [2, 3, 4, 5, 6]; ci = R['cluster_indices']; sil = [ci[str(k)]['sil'] for k in ks]; wss = [ci[str(k)]['WSS'] for k in ks]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
a1.plot(ks, sil, 'o-', color='#2b6cb0', lw=2); a2.plot(ks, wss, 's-', color='#2e8b57', lw=2)
for a, v, f in [(a1, sil, '{:.3f}'), (a2, wss, '{:.2f}')]:
    a.axvline(3, ls='--', color='k', lw=1); a.set_xticks(ks); a.set_xlabel('Number of clusters (k)'); a.spines[['top', 'right']].set_visible(False)
    for kk, y in zip(ks, v): a.annotate(f.format(y), (kk, y), textcoords='offset points', xytext=(10, 6), ha='left', fontsize=9)
a1.set_ylabel('Average silhouette coefficient'); a2.set_ylabel('Within-cluster sum of squares')
a1.text(-0.12, 1.04, '(a)', transform=a1.transAxes, fontweight='bold'); a2.text(-0.12, 1.04, '(b)', transform=a2.transAxes, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{FIG}/FigureS1.png', dpi=600); plt.close()
# Figure S2
co = pd.read_csv(f'{OUT}/coassign.csv', index_col=0).loc[T, T]
ordr = sorted(T, key=lambda t: (CLU[t], T.index(t)))
fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(co.loc[ordr, ordr].values, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(10), [NM[t] for t in ordr], rotation=45, ha='right'); ax.set_yticks(range(10), [NM[t] for t in ordr])
for i in range(10):
    for j in range(10):
        v = co.loc[ordr[i], ordr[j]]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > 0.6 else 'black')
plt.colorbar(im, fraction=0.046, pad=0.04, label='Co-assignment proportion'); plt.tight_layout(); plt.savefig(f'{FIG}/FigureS2.png', dpi=600); plt.close()
# Figure 4
mw = pd.read_csv(f'{OUT}/table6_MW.csv')
fig, ax = plt.subplots(2, 3, figsize=(15, 8.6), sharey='row'); x = np.arange(6); w = 0.36
for j, c in enumerate([1, 2, 3]):
    s = mw[mw.cluster == c].set_index('attr').loc[A]
    mk = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ('†' if pr < 0.05 else '') for p, pr in zip(s.p_bonf, s.p)]
    a0 = ax[0, j]; a0.bar(x - w / 2, s.meanD, w, color='#8c8c8c', label='Domestic-language reviews'); a0.bar(x + w / 2, s.meanF, w, color='#222222', label='Non-domestic-language reviews')
    for i, t in enumerate(mk):
        if t: a0.text(x[i], max(s.meanD.iloc[i], s.meanF.iloc[i], 0) + 0.08, t, ha='center', va='bottom', fontsize=12, fontweight='bold')
    a0.axhline(0, color='k', lw=0.8); a0.set_ylim(-1.8, 2.3); a0.set_xticks(x, AS)
    mem = ' · '.join(NM[t] for t in T if CLU[t] == c and t != 'Eurostar'); a0.set_title(f'{TITLE[c]}\n({mem})', fontsize=11, fontweight='bold')
    a0.text(0.02, 0.96, '(a)', transform=a0.transAxes, fontweight='bold', va='top')
    a1 = ax[1, j]; a1.bar(x - w / 2, s.MR_D, w, color='#8c8c8c'); a1.bar(x + w / 2, s.MR_F, w, color='#222222'); a1.set_ylim(0, 105); a1.set_xticks(x, AS)
    a1.text(0.02, 0.96, '(b)', transform=a1.transAxes, fontweight='bold', va='top')
    for aa in (a0, a1): aa.spines[['top', 'right']].set_visible(False)
ax[0, 0].set_ylabel('Mean sentiment score (−2 to +2)'); ax[1, 0].set_ylabel('Mention ratio (%)')
hh, ll = ax[0, 0].get_legend_handles_labels(); fig.legend(hh, ll, loc='lower center', ncol=2, frameon=False, fontsize=11)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig(f'{FIG}/Figure4.png', dpi=600); plt.close()
# Figure 5 and 6 (if keyword results exist)
if os.path.exists(f'{OUT}/table7_Za.csv'):
    za = pd.read_csv(f'{OUT}/table7_Za.csv').pivot(index='attr', columns='cluster', values='Z_a').loc[A]
    fig, ax = plt.subplots(figsize=(9, 4.8)); w = 0.26
    for i, c in enumerate([1, 2, 3]): ax.bar(np.arange(6) + (i - 1) * w, za[c], w, color=CC[c], label=TITLE[c])
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(range(6), AS); ax.set_ylabel('Attribute-level salience $Z_a$'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(f'{FIG}/Figure5.png', dpi=600); plt.close()
    kw = pd.read_csv(f'{OUT}/keywords.csv'); kw = kw[(kw.translated_only == False) & kw.group.isin(['F', 'D'])]
    OI = {'Tangibility': '#0072B2', 'Reliability': '#D55E00', 'Responsiveness': '#009E73', 'Assurance': '#CC79A7', 'Empathy': '#E69F00', 'Comfort': '#56B4E9'}
    fig, axs = plt.subplots(1, 3, figsize=(16, 7.5))
    for i, c in enumerate([1, 2, 3]):
        a = axs[i]; f = kw[(kw.cluster == c) & (kw.group == 'F')].head(15); dd = kw[(kw.cluster == c) & (kw.group == 'D')].head(15)
        y = np.arange(15)
        a.barh(y, f.z.values, color=[OI.get(v, '#bbbbbb') if isinstance(v, str) else '#bbbbbb' for v in f.attr.fillna('')])
        a.barh(y, dd.z.values, color=[OI.get(v, '#bbbbbb') if isinstance(v, str) else '#bbbbbb' for v in dd.attr.fillna('')])
        for yy, (wd, z) in enumerate(zip(f.word, f.z)): a.text(z + 0.2, yy, wd, va='center', fontsize=9)
        for yy, (wd, z) in enumerate(zip(dd.word, dd.z)): a.text(z - 0.2, yy, wd, va='center', ha='right', fontsize=9)
        lim = max(abs(kw[kw.cluster == c].z).max() * 1.45, 5); a.set_xlim(-lim, lim); a.invert_yaxis(); a.set_yticks([]); a.axvline(0, color='k', lw=0.8)
        a.set_title(TITLE[c], fontweight='bold', color=CC[c]); a.set_xlabel('$Z_w$  (← domestic-language | non-domestic-language →)'); a.spines[['top', 'right', 'left']].set_visible(False)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=v) for v in OI.values()] + [Patch(color='#bbbbbb')], list(OI) + ['None'], loc='lower center', ncol=7, frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(f'{FIG}/Figure6.png', dpi=600); plt.close()
json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, ensure_ascii=False)
print('figures saved', sorted(os.listdir(FIG)))
