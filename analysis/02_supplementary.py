"""Supplementary analyses on the re-annotated corpus: Chinese script sensitivity, error simulation,
alternative partitions, speed bootstrap CI, boundary distances.  Usage (from the repository root): python analysis/02_supplementary.py data/claude_sonnet46_annotations.csv results"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import rankdata, norm, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings('ignore')
ANN, OUT = sys.argv[1], sys.argv[2]
A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']
T = ['KTX', 'Shinkansen', 'THSR', 'ICE', 'Eurostar', 'Frecciarossa', 'Italo', 'TGV', 'RENFE', 'CRH']
DOM = {'KTX': 'ko', 'Shinkansen': 'ja', 'THSR': 'zh', 'ICE': 'de', 'Frecciarossa': 'it', 'Italo': 'it', 'TGV': 'fr', 'RENFE': 'es', 'CRH': 'zh'}
SPEED = {'KTX': 305, 'Shinkansen': 285, 'THSR': 300, 'ICE': 300, 'Eurostar': 300, 'Frecciarossa': 300, 'Italo': 300, 'TGV': 300, 'RENFE': 300, 'CRH': 350}
R0 = json.load(open(f'{OUT}/results.json')); REF = R0['clusters']
d = pd.read_csv(ANN, encoding='utf-8-sig')
def lab(v):
    v = str(v).lower()
    if 'not' in v or v in ('nan', 'none', ''): return 'N'
    return 'M' if ('mix' in v) else ('Z' if not ('pos' in v or 'neg' in v) else 'S')
M = np.zeros((len(d), 6), bool); X = np.zeros((len(d), 6))
for j, a in enumerate(A):
    l = d[a + '_sentiment'].map(lab); M[:, j] = l != 'N'
    s = pd.to_numeric(d[a + '_score'], errors='coerce').fillna(0).clip(-2, 2).to_numpy(copy=True); s[l.isin(['M', 'Z']).values] = 0; X[:, j] = s
sysv = d.train_name.values; lang = d.language.values
isdom = np.array([DOM.get(t) == l for t, l in zip(sysv, lang)])
ref = np.array([REF[t] for t in T])
O = {}
def SSmat(Mx, Xx):
    return np.array([[Xx[(sysv == t) & Mx[:, j], j].mean() for j in range(6)] for t in T])
def clus(S, k=3): return fcluster(linkage(StandardScaler().fit_transform(S), 'complete'), k, 'maxclust')
def vanel(Mx, Xx, systems, j, dom=isdom):
    num = var = 0
    for t in systems:
        s = (sysv == t) & Mx[:, j]; f = Xx[s & ~dom, j]; dm = Xx[s & dom, j]
        if len(f) == 0 or len(dm) == 0: continue
        n1, n2 = len(f), len(dm); N = n1 + n2; w = 1 / (N + 1); allv = np.concatenate([f, dm]); rk = rankdata(allv)
        ties = np.unique(allv, return_counts=True)[1]; v = n1 * n2 / 12 * ((N + 1) - (ties ** 3 - ties).sum() / (N * (N - 1)))
        num += w * (rk[:n1].sum() - n1 * (N + 1) / 2); var += w * w * v
    z = num / np.sqrt(var); return z, 2 * norm.sf(abs(z))
GR = {c: [t for t in T if REF[t] == c and t in DOM] for c in [1, 2, 3]}
def nsig(Mx, Xx, groups=GR, dom=isdom):
    r = [vanel(Mx, Xx, groups[c], j, dom) for c in groups for j in range(6)]
    m = len(r); return sum(1 for z, p in r if p * m < 0.05 and z > 0), sum(1 for z, p in r if p * m < 0.05 and z < 0)
def cl(systems, j, dom=isdom):
    s = pd.DataFrame({'y': X[:, j], 'm': M[:, j], 't': sysv, 'f': ~dom}); s = s[s.m & s.t.isin(systems)]
    Xd = pd.get_dummies(s.t, drop_first=True).astype(float); Xd['foreign'] = s.f.astype(float)
    r = OrderedModel(s.y.astype(int).values, Xd.values, distr='logit').fit(method='bfgs', disp=False, maxiter=3000)
    k = list(Xd.columns).index('foreign'); return np.exp(r.params[k]), r.pvalues[k]
# 1. Chinese script sensitivity (script of the original Chinese text)
# script of the original Chinese text (simp/trad), precomputed with OpenCC because review texts are not redistributed
zs = pd.read_csv('data/chinese_script.csv'); scr = dict(zip(zip(zs.train_name, zs.review_id), zs.script))
sc = np.array([scr.get((t, r), '') if l == 'zh' else '' for t, r, l in zip(sysv, d.review_id, lang)])
O['zh_script_crosstab'] = pd.crosstab(sysv[lang == 'zh'], sc[lang == 'zh']).to_dict('index')
dom1 = isdom.copy(); dom1[(sysv == 'THSR') & (lang == 'zh') & (sc == 'simp')] = False; dom1[(sysv == 'CRH') & (lang == 'zh') & (sc == 'trad')] = False
O['zh_moved'] = int((dom1 != isdom).sum())
rows = []
for c in [1, 2, 3]:
    for j, a in enumerate(A):
        o0, p0 = cl(GR[c], j); o1, p1 = cl(GR[c], j, dom1)
        rows.append(dict(cluster=c, attr=a, OR_base=o0, p_base=min(1, 18 * p0), OR_script=o1, p_script=min(1, 18 * p1)))
zs = pd.DataFrame(rows); zs.to_csv(f'{OUT}/S11_script.csv', index=False)
O['zh_sig_base_script'] = [int(((zs.p_base < .05) & (zs.OR_base > 1)).sum()), int(((zs.p_script < .05) & (zs.OR_script > 1)).sum())]
print('script done', O['zh_moved'], O['zh_sig_base_script'], flush=True)
# 2. alternative partitions k = 2 and k = 4 (system-adjusted CL)
alt = []
for k in [2, 4]:
    lb = clus(SSmat(M, X), k); grp = {g: [t for t, l in zip(T, lb) if l == g and t in DOM] for g in np.unique(lb)}
    for g, sy in grp.items():
        if len(sy) == 0: continue
        for j, a in enumerate(A):
            o, p = cl(sy, j) if len(sy) > 1 else (np.nan, np.nan)
            if len(sy) == 1:  # single system: no system effect needed
                s = pd.DataFrame({'y': X[:, j], 'm': M[:, j], 't': sysv, 'f': ~isdom}); s = s[s.m & (s.t == sy[0])]
                r = OrderedModel(s.y.astype(int).values, s[['f']].astype(float).values, distr='logit').fit(method='bfgs', disp=False, maxiter=3000)
                o, p = np.exp(r.params[0]), r.pvalues[0]
            alt.append(dict(k=k, group='+'.join(sy), attr=a, OR=o, p_bonf=min(1, p * 6 * len(grp))))
alt = pd.DataFrame(alt); alt.to_csv(f'{OUT}/S10_alt_partitions.csv', index=False)
O['alt_sig'] = {int(k): [int(((g.p_bonf < .05) & (g.OR > 1)).sum()), int(((g.p_bonf < .05) & (g.OR < 1)).sum()), len(g)] for k, g in alt.groupby('k')}
print('alt done', O['alt_sig'], flush=True)
# 3. boundary distances (standardized SS) for Italo and CRH
Z = StandardScaler().fit_transform(SSmat(M, X)); cen = {c: Z[ref == c].mean(0) for c in [1, 2, 3]}
O['dist'] = {t: {c: round(float(np.linalg.norm(Z[T.index(t)] - cen[c])), 2) for c in [1, 2, 3]} for t in ['Italo', 'CRH', 'ICE', 'RENFE']}
# 4. speed bootstrap CI over systems
rng = np.random.default_rng(42); ms = SSmat(M, X).mean(1); sp = np.array([SPEED[t] for t in T]); bs = []
for b in range(2000):
    i = rng.integers(0, 10, 10)
    if len(np.unique(sp[i])) > 1 and len(np.unique(ms[i])) > 1: bs.append(spearmanr(sp[i], ms[i])[0])
O['speed_CI'] = [round(float(np.nanpercentile(bs, 2.5)), 2), round(float(np.nanpercentile(bs, 97.5)), 2)]
# 5. error simulation
O['VE_base'] = nsig(M, X)
rng = np.random.default_rng(2026); sim = []
for e in [0.05, 0.10, 0.20, 0.30]:
    for scen in ['random', 'miss', 'dom_plus1']:
        keep = 0; aris = []; nf = []; nd = []
        for r in range(200):
            Mx = M.copy(); Xx = X.copy()
            if scen == 'random':
                u = (rng.random(M.shape) < e) & M; Xx[u] = rng.integers(-2, 3, u.sum())
            elif scen == 'miss':
                u = (rng.random(M.shape) < e) & M; Mx[u] = False
            else:
                u = (rng.random(M.shape) < e) & M & isdom[:, None]; Xx[u] = np.minimum(Xx[u] + 1, 2)
            lb = clus(SSmat(Mx, Xx)); a_ = adjusted_rand_score(ref, lb); aris.append(a_); keep += a_ == 1
            if r < 50:
                f_, d_ = nsig(Mx, Xx); nf.append(f_); nd.append(d_)
        sim.append(dict(error=e, scenario=scen, partition_kept=keep / 200, mean_ARI=np.mean(aris), sig_foreign_mean=np.mean(nf), sig_foreign_min=np.min(nf), sig_dom_mean=np.mean(nd)))
        print(sim[-1], flush=True)
pd.DataFrame(sim).to_csv(f'{OUT}/S8_error_sim.csv', index=False)
# tipping point: domestic-language scores underestimated by one point; and overestimated (reverse)
tip = []
for e in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    nf = []; nd = []
    for r in range(20):
        Xx = X.copy(); u = (rng.random(M.shape) < e) & M & isdom[:, None]; Xx[u] = np.minimum(Xx[u] + 1, 2); f_, d_ = nsig(M, Xx); nf.append(f_); nd.append(d_)
    tip.append(dict(error=e, sig_foreign_mean=np.mean(nf), sig_dom_mean=np.mean(nd))); print(tip[-1], flush=True)
pd.DataFrame(tip).to_csv(f'{OUT}/S8_tipping.csv', index=False)
json.dump(O, open(f'{OUT}/extra.json', 'w'), indent=1, default=str)
print(json.dumps(O, indent=1, default=str))
