"""Full reanalysis on the re-annotated corpus (original texts).
Usage (from the repository root): python analysis/01_reanalysis.py data/claude_sonnet46_annotations.csv results"""
import sys, os, json, re, glob, warnings
import numpy as np, pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import mannwhitneyu, rankdata, norm, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings('ignore')
ANN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']
T = ['KTX', 'Shinkansen', 'THSR', 'ICE', 'Eurostar', 'Frecciarossa', 'Italo', 'TGV', 'RENFE', 'CRH']
DOM = {'KTX': 'ko', 'Shinkansen': 'ja', 'THSR': 'zh', 'ICE': 'de', 'Frecciarossa': 'it', 'Italo': 'it', 'TGV': 'fr', 'RENFE': 'es', 'CRH': 'zh'}
SPEED = {'KTX': 305, 'Shinkansen': 285, 'THSR': 300, 'ICE': 300, 'Eurostar': 300, 'Frecciarossa': 300, 'Italo': 300, 'TGV': 300, 'RENFE': 300, 'CRH': 350}
R = {}
def lab(v):
    v = str(v).lower()
    if 'not' in v or v in ('nan', 'none', ''): return 'Not_mentioned'
    if 'pos' in v: return 'Positive'
    if 'neg' in v: return 'Negative'
    if 'mix' in v: return 'Mixed'
    return 'Neutral'
d = pd.read_csv(ANN, encoding='utf-8-sig')
for a in A:
    d[a + '_lab'] = d[a + '_sentiment'].map(lab)
    d[a + '_m'] = d[a + '_lab'] != 'Not_mentioned'
    s = pd.to_numeric(d[a + '_score'], errors='coerce').fillna(0).clip(-2, 2)
    s[d[a + '_lab'].isin(['Mixed', 'Neutral'])] = 0
    d[a + '_s'] = s.where(d[a + '_m'])
R['n_reviews'] = len(d); R['n_annotated'] = int(d.annotated.sum()) if 'annotated' in d else len(d)
R['label_counts'] = {a: d[a + '_lab'].value_counts().to_dict() for a in A}
# ---------- Table 3
MR = pd.DataFrame({a: [100 * d.loc[d.train_name == t, a + '_m'].mean() for t in T] for a in A}, index=T)
SS = pd.DataFrame({a: [d.loc[(d.train_name == t) & d[a + '_m'], a + '_s'].mean() for t in T] for a in A}, index=T)
NN = pd.DataFrame({a: [int(d.loc[(d.train_name == t) & d[a + '_m']].shape[0]) for t in T] for a in A}, index=T)
SD = pd.DataFrame({a: [d.loc[(d.train_name == t) & d[a + '_m'], a + '_s'].std() for t in T] for a in A}, index=T)
MR.round(1).to_csv(f'{OUT}/table3_MR.csv'); SS.round(3).to_csv(f'{OUT}/table3_SS.csv'); NN.to_csv(f'{OUT}/table3_n.csv')
(1.96 * SD / np.sqrt(NN)).round(3).to_csv(f'{OUT}/tableS1_CIhalf.csv')
R['MR_mean'] = MR.mean().round(1).to_dict(); R['MR_range'] = {a: [round(MR[a].min(), 1), round(MR[a].max(), 1)] for a in A}
# ---------- clustering
X = StandardScaler().fit_transform(SS.values)
Lc = linkage(X, 'complete')
idx = {}
rng = np.random.default_rng(42)
lo, hi = X.min(0), X.max(0)
def wss(Z, lab): return sum(((Z[lab == g] - Z[lab == g].mean(0)) ** 2).sum() for g in np.unique(lab))
for k in range(2, 7):
    lab = fcluster(Lc, k, 'maxclust')
    idx[k] = {'sil': silhouette_score(X, lab), 'WSS': wss(X, lab), 'CH': calinski_harabasz_score(X, lab)}
for k in range(1, 7):
    lab = fcluster(Lc, k, 'maxclust') if k > 1 else np.ones(10, int)
    lw = np.log(wss(X, lab)); ref = []
    for b in range(100):
        Zr = rng.uniform(lo, hi, X.shape); lr = fcluster(linkage(Zr, 'complete'), k, 'maxclust') if k > 1 else np.ones(10, int)
        ref.append(np.log(wss(Zr, lr)))
    idx.setdefault(k, {}).update({'gap': float(np.mean(ref) - lw), 'sk': float(np.std(ref))})
R['cluster_indices'] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in idx.items()}
gap_k = next((k for k in range(1, 6) if idx[k]['gap'] >= idx[k + 1]['gap'] - idx[k + 1]['sk']), None)
R['gap_selected_k'] = gap_k
R['merge_heights'] = [round(float(h), 3) for h in Lc[:, 2]]
def clus(S, k=3, method='complete'):
    return fcluster(linkage(StandardScaler().fit_transform(S), method), k, 'maxclust')
for k in [2, 3, 4]:
    lab = clus(SS.values, k); R[f'partition_k{k}'] = [[t for t, l in zip(T, lab) if l == g] for g in np.unique(lab)]
lab3 = clus(SS.values, 3)
# order clusters by mean SS (1 = highest)
cm = {g: SS.values[lab3 == g].mean() for g in np.unique(lab3)}
srt = sorted(cm, key=lambda g: -cm[g]); order = {srt[0]: 1, srt[-1]: 2, srt[1]: 3}  # 1 high, 2 low, 3 intermediate (as in the manuscript)
CLU = {t: order[l] for t, l in zip(T, lab3)}
R['clusters'] = CLU
ref = np.array([CLU[t] for t in T])
R['cluster_profile'] = {c: SS.loc[[t for t in T if CLU[t] == c]].mean().round(3).to_dict() for c in [1, 2, 3]}
R['cluster_MR'] = {c: MR.loc[[t for t in T if CLU[t] == c]].mean().round(1).to_dict() for c in [1, 2, 3]}
R['linkage'] = {m: round(adjusted_rand_score(ref, clus(SS.values, 3, m)), 3) for m in ['average', 'single', 'ward']}
R['linkage_groups'] = {m: [[t for t, l in zip(T, clus(SS.values, 3, m)) if l == g] for g in np.unique(clus(SS.values, 3, m))] for m in ['average', 'single', 'ward']}
T9 = [t for t in T if t != 'Eurostar']; R['noEurostar_ARI'] = round(adjusted_rand_score([CLU[t] for t in T9], clus(SS.loc[T9].values, 3)), 3)
R['noEurostar_sil'] = round(silhouette_score(StandardScaler().fit_transform(SS.loc[T9].values), clus(SS.loc[T9].values, 3)), 3)
# bootstrap
rngb = np.random.default_rng(2026); B = 1000; co = np.zeros((10, 10)); same = 0; jac = {1: [], 2: [], 3: []}
S_arr = {t: {a: d.loc[(d.train_name == t) & d[a + '_m'], a + '_s'].values for a in A} for t in T}
refsets = {c: set(np.where(ref == c)[0]) for c in [1, 2, 3]}
for b in range(B):
    M = np.array([[rngb.choice(S_arr[t][a], len(S_arr[t][a])).mean() for a in A] for t in T])
    lb = clus(M); co += (lb[:, None] == lb[None, :]); same += adjusted_rand_score(ref, lb) == 1
    bs = [set(np.where(lb == g)[0]) for g in np.unique(lb)]
    for c in [1, 2, 3]: jac[c].append(max(len(refsets[c] & s) / len(refsets[c] | s) for s in bs))
co /= B; pd.DataFrame(co, index=T, columns=T).round(3).to_csv(f'{OUT}/coassign.csv')
R['boot_same_partition'] = same / B; R['boot_jaccard'] = {c: round(float(np.mean(v)), 3) for c, v in jac.items()}
# PCA
p = PCA().fit(X); R['pca_var'] = p.explained_variance_ratio_.round(4).tolist(); R['pc1_loadings'] = p.components_[0].round(3).tolist()
R['corr_range'] = [round(float(np.min(np.corrcoef(X.T)[np.triu_indices(6, 1)])), 3), round(float(np.max(np.corrcoef(X.T)[np.triu_indices(6, 1)])), 3)]
pc1 = p.transform(X)[:, :1]; R['pc1_partition_ARI'] = round(adjusted_rand_score(ref, fcluster(linkage(pc1, 'complete'), 3, 'maxclust')), 3)
resid = X - np.outer(p.transform(X)[:, 0], p.components_[0]); R['pc1_residuals'] = pd.DataFrame(resid, index=T, columns=A).round(2).to_dict('index')
m = SS.mean(1); r_, p_ = spearmanr([SPEED[t] for t in T], m); R['speed_spearman'] = [round(r_, 3), round(p_, 3)]
R['mean_SS'] = m.round(3).to_dict()
# NSI
NSI = pd.DataFrame({a: [d.loc[d.train_name == t, a + '_s'].fillna(0).sum() / (d.train_name == t).sum() for t in T] for a in A}, index=T)
R['NSI_ARI'] = round(adjusted_rand_score(ref, clus(NSI.values)), 3); R['NSI_spearman'] = {a: round(spearmanr(SS[a], NSI[a])[0], 3) for a in A}
NSI.round(3).to_csv(f'{OUT}/S9_NSI.csv')
# ---------- domestic vs non-domestic
d['inD'] = d.train_name.isin(DOM.keys()); d['dom'] = [DOM.get(t) == l for t, l in zip(d.train_name, d.language)]
d['cl'] = d.train_name.map(CLU)
dd = d[d.inD]
rows, strat, per = [], [], []
def vanel(sub, a):
    num = var = 0
    for t, g in sub.groupby('train_name'):
        f = g[~g.dom][a + '_s'].values; dm = g[g.dom][a + '_s'].values
        if len(f) == 0 or len(dm) == 0: continue
        n1, n2 = len(f), len(dm); N = n1 + n2; w = 1 / (N + 1); allv = np.concatenate([f, dm]); rk = rankdata(allv)
        ties = np.unique(allv, return_counts=True)[1]; v = n1 * n2 / 12 * ((N + 1) - (ties ** 3 - ties).sum() / (N * (N - 1)))
        num += w * (rk[:n1].sum() - n1 * (N + 1) / 2); var += w * w * v
    z = num / np.sqrt(var); return z, 2 * norm.sf(abs(z))
def cl_model(sub, a, extra=None):
    y = sub[a + '_s'].astype(int); Xd = pd.get_dummies(sub.train_name, drop_first=True).astype(float)
    if extra is not None: Xd = pd.concat([Xd, extra.loc[sub.index]], axis=1)
    Xd['foreign'] = (~sub.dom).astype(float)
    r = OrderedModel(y.values, Xd.values, distr='logit').fit(method='bfgs', disp=False, maxiter=3000)
    j = list(Xd.columns).index('foreign'); return r.params[j], r.bse[j], r.pvalues[j]
for c in [1, 2, 3]:
    for a in A:
        s = dd[(dd.cl == c) & dd[a + '_m']]
        f = s[~s.dom][a + '_s']; dm = s[s.dom][a + '_s']
        U, pp = mannwhitneyu(f, dm, alternative='two-sided')
        rows.append(dict(cluster=c, attr=a, nD=len(dm), nF=len(f), meanD=dm.mean(), meanF=f.mean(), U=U, p=pp, p_bonf=min(1, 18 * pp), r_rb=1 - 2 * U / (len(f) * len(dm)),
                         MR_D=100 * dd[(dd.cl == c) & dd.dom][a + '_m'].mean(), MR_F=100 * dd[(dd.cl == c) & ~dd.dom][a + '_m'].mean()))
        z, pv = vanel(s, a); b, se, po = cl_model(s, a)
        strat.append(dict(cluster=c, attr=a, z_VE=z, p_VE=pv, p_VE_bonf=min(1, 18 * pv), OR=np.exp(b), OR_lo=np.exp(b - 1.96 * se), OR_hi=np.exp(b + 1.96 * se), p_CL=po, p_CL_bonf=min(1, 18 * po), z_CL=b / se))
        for t, g in s.groupby('train_name'):
            f2 = g[~g.dom][a + '_s']; d2 = g[g.dom][a + '_s']
            if len(f2) and len(d2):
                U2, p2 = mannwhitneyu(f2, d2, alternative='two-sided')
                per.append(dict(cluster=c, system=t, attr=a, nD=len(d2), nF=len(f2), meanD=d2.mean(), meanF=f2.mean(), p=p2, p_bonf54=min(1, 54 * p2), r_rb=1 - 2 * U2 / (len(f2) * len(d2))))
mw = pd.DataFrame(rows); st = pd.DataFrame(strat); ps = pd.DataFrame(per)
mw.to_csv(f'{OUT}/table6_MW.csv', index=False); st.to_csv(f'{OUT}/table6_CL_VE.csv', index=False); ps.to_csv(f'{OUT}/S3_per_system.csv', index=False)
R['n_groups'] = {c: [int(dd[(dd.cl == c) & dd.dom].shape[0]), int(dd[(dd.cl == c) & ~dd.dom].shape[0])] for c in [1, 2, 3]}
R['MW_sig'] = int((mw.p_bonf < 0.05).sum()); R['MW_sig_foreign'] = int(((mw.p_bonf < 0.05) & (mw.r_rb < 0)).sum())
pv = np.sort(mw.p.values); R['Holm_sig'] = sum(1 for i in range(18) if all(pv[j] <= 0.05 / (18 - j) for j in range(i + 1)))
R['CL_sig_foreign'] = int(((st.p_CL_bonf < 0.05) & (st.OR > 1)).sum()); R['CL_sig_domestic'] = int(((st.p_CL_bonf < 0.05) & (st.OR < 1)).sum())
R['VE_sig_foreign'] = int(((st.p_VE_bonf < 0.05) & (st.z_VE > 0)).sum()); R['VE_sig_domestic'] = int(((st.p_VE_bonf < 0.05) & (st.z_VE < 0)).sum())
sigOR = st[(st.p_CL_bonf < 0.05)].OR; R['CL_OR_range_sig'] = [round(sigOR.min(), 2), round(sigOR.max(), 2)] if len(sigOR) else None
R['rrb_range'] = [round(mw.r_rb.min(), 3), round(mw.r_rb.max(), 3)]
R['per_system_sig_domestic'] = int(((ps.p_bonf54 < 0.05) & (ps.r_rb > 0)).sum()); R['per_system_sig_foreign'] = int(((ps.p_bonf54 < 0.05) & (ps.r_rb < 0)).sum())
R['per_system_domestic_higher'] = int((ps.r_rb > 0).sum())
# English vs other-language foreign
L4 = []
for c in [1, 2, 3]:
    for a in A:
        s = dd[(dd.cl == c) & dd[a + '_m']]
        out = {'cluster': c, 'attr': a}
        for nm, sel in [('EN', s.language == 'en'), ('OT', s.language != 'en')]:
            sub = s[s.dom | (~s.dom & sel)]
            try: b, se, po = cl_model(sub, a); out[f'OR_{nm}'] = np.exp(b); out[f'p_{nm}'] = min(1, 18 * po)
            except Exception: out[f'OR_{nm}'] = np.nan; out[f'p_{nm}'] = np.nan
        out['n_OT'] = int((~s.dom & (s.language != 'en')).sum()); L4.append(out)
L4 = pd.DataFrame(L4); L4.to_csv(f'{OUT}/S4_lang.csv', index=False)
R['S4'] = {nm: [int((L4[f'OR_{nm}'] > 1).sum()), int(((L4[f'p_{nm}'] < 0.05) & (L4[f'OR_{nm}'] > 1)).sum()), int(((L4[f'p_{nm}'] < 0.05) & (L4[f'OR_{nm}'] < 1)).sum())] for nm in ['EN', 'OT']}
# design effect
zc = st.z_CL.values; R['deff'] = {D: int(((2 * norm.sf(np.abs(zc) / np.sqrt(D)) * 18 < 0.05) & (zc > 0)).sum()) for D in [1, 1.5, 2, 3]}
# covariates: trip type and season (if available)
meta = 'data/review_metadata.csv'
if os.path.exists(meta):
    me = pd.read_csv(meta); d2 = d.merge(me, on=['train_name', 'review_id'], how='left')
    d2['trip'] = d2.trip_type.fillna('unknown')
    d2['vyear'] = pd.to_numeric(d2.visit_month.astype(str).str[:4], errors='coerce')
    d2['period'] = pd.cut(d2.vyear, [0, 2019, 2021, 2100], labels=['<=2019', '2020-21', '>=2022']).astype(str)
    d2['review_mean'] = d2[[a + '_s' for a in A]].mean(axis=1)
    R['star_spearman_all'] = round(spearmanr(d2.review_mean, d2.star, nan_policy='omit')[0], 3)
    R['star_spearman_by_system'] = {t: round(spearmanr(g.review_mean, g.star, nan_policy='omit')[0], 3) for t, g in d2.groupby('train_name')}
    R['star_spearman_by_lang'] = {l: round(spearmanr(g.review_mean, g.star, nan_policy='omit')[0], 3) for l, g in d2.groupby('language') if len(g) >= 200}
    R['star_attr_spearman'] = {a: round(spearmanr(d2[a + '_s'], d2.star, nan_policy='omit')[0], 3) for a in A}
    # CL with trip type + period covariates
    d2 = d2.set_index(d.index); cov = pd.get_dummies(d2[['trip', 'period']], drop_first=True).astype(float)
    d['dom'] = d['dom']; covst = []
    for c in [1, 2, 3]:
        for a in A:
            s = d[(d.cl == c) & d[a + '_m'] & d.inD]
            b, se, po = cl_model(s, a, cov); covst.append(dict(cluster=c, attr=a, OR=np.exp(b), p_bonf=min(1, 18 * po)))
    covst = pd.DataFrame(covst); covst.to_csv(f'{OUT}/S_cov_trip_period.csv', index=False)
    R['CL_cov_sig_foreign'] = int(((covst.p_bonf < 0.05) & (covst.OR > 1)).sum()); R['CL_cov_sig_domestic'] = int(((covst.p_bonf < 0.05) & (covst.OR < 1)).sum())
    R['trip_dist'] = d2.trip.value_counts().to_dict(); R['period_dist'] = d2.period.value_counts().to_dict()
    R['written_range'] = [str(pd.to_datetime(d2.written).min())[:10], str(pd.to_datetime(d2.written).max())[:10]]
# ---------- selection tone
tone = []
for t in T:
    s = d[d.train_name == t]
    for a in A:
        oth = [x for x in A if x != a]; mo = s[[x + '_s' for x in oth]].mean(axis=1)
        tone.append(dict(system=t, attr=a, diff=mo[s[a + '_m']].mean() - mo[~s[a + '_m']].mean()))
tone = pd.DataFrame(tone); R['tone_diff'] = tone.groupby('attr')['diff'].agg(['mean', 'min', 'max']).round(3).to_dict('index')
multi = d[[a + '_m' for a in A]].sum(axis=1) >= 2
R['multi_attr_n'] = int(multi.sum()); R['multi_attr_diff_scores_pct'] = round(100 * d[multi][[a + '_s' for a in A]].apply(lambda r: r.dropna().nunique() > 1, axis=1).mean(), 1)
json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o), ensure_ascii=False)
print(json.dumps({k: R[k] for k in ['clusters', 'cluster_indices', 'gap_selected_k', 'boot_same_partition', 'boot_jaccard', 'pca_var', 'MW_sig', 'CL_sig_foreign', 'CL_sig_domestic', 'VE_sig_foreign', 'S4', 'deff', 'speed_spearman'] if k in R}, indent=1, default=str))
