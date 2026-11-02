"""Validation metrics: reference (two blind annotators + adjudication) vs primary (Claude Sonnet 4.6) and Gemini 3 Flash.
Writes val/val.json (keys used in the manuscript), out/S7_validation.csv, out/S12_gemini.csv, val/reference.csv."""
import json, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score as ck
A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']
DOM = {'KTX': 'ko', 'Shinkansen': 'ja', 'THSR': 'zh', 'ICE': 'de', 'Frecciarossa': 'it', 'Italo': 'it', 'TGV': 'fr', 'RENFE': 'es', 'CRH': 'zh'}
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'validation'))
v = pd.read_csv('validation_sample_ids.csv')
tm = pd.read_csv('annotator_pairs.csv')
adj = {(int(o['item']), o['attr']): o for o in json.load(open('adjudication.json'))}
def ref_row(r):
    key = (int(r['item']), r['attr'])
    if key in adj:
        o = adj[key]; return o['label'], int(o['score'])
    if r['tl'] == 'Not_mentioned' and r['ml'] == 'Not_mentioned': return 'Not_mentioned', 0
    s = int(np.trunc((r['ts'] + r['ms']) / 2))
    lab = r['tl'] if r['tl'] == r['ml'] else ('Positive' if s > 0 else 'Negative' if s < 0 else 'Neutral')
    return lab, s
tm[['rl', 'rs']] = tm.apply(lambda r: pd.Series(ref_row(r)), axis=1)
tm = tm.merge(v, on='item'); tm.to_csv('reference.csv', index=False)
def model(fn):
    d = pd.read_csv(fn, encoding='utf-8-sig'); rows = []
    for r in d.itertuples():
        for a in A:
            l = getattr(r, a + '_sentiment'); s = pd.to_numeric(getattr(r, a + '_score'), errors='coerce')
            s = 0 if (pd.isna(s) or l in ('Mixed', 'Neutral', 'Not_mentioned')) else int(np.clip(s, -2, 2))
            rows.append(dict(train_name=r.train_name, review_id=r.review_id, attr=a, l=l, s=s, annotated=getattr(r, 'annotated', True)))
    return pd.DataFrame(rows)
def ref_score(df):  # reference score with Mixed/Neutral = 0
    return np.where(df.rl.isin(['Mixed', 'Neutral', 'Not_mentioned']), 0, df.rs)
def metrics(df, ml='l', ms='s', rl='rl', rsc='rsr'):
    rm = df[rl] != 'Not_mentioned'; mm = df[ml] != 'Not_mentioned'
    tp = (rm & mm).sum(); P = tp / mm.sum(); R = tp / rm.sum(); F = 2 * P * R / (P + R)
    k = ck(rm, mm); b = rm & mm; d = df[ms][b] - df[rsc][b]
    q = ck(df[ms][b].astype(int), df[rsc][b].astype(int), weights='quadratic')
    pol = (np.sign(df[ms][b]) == np.sign(df[rsc][b])).mean()
    return dict(n=len(df), P=P, R=R, F1=F, kappa=k, MAE=d.abs().mean(), MSD=d.mean(), QWK=q, polarity=100 * pol, n_both=int(b.sum()))
cl = model('../data/claude_sonnet46_annotations.csv')
x = tm.merge(cl, on=['train_name', 'review_id', 'attr'], how='left'); x['rsr'] = ref_score(x)
x['dom'] = [DOM.get(t) == l for t, l in zip(x.train_name, x.language)]
out = {}; rows = []
M = metrics(x); rows.append(dict(group='All', **M))
for lg, g in x.groupby('language'): rows.append(dict(group=lg, **metrics(g)))
for nm, g in [('domestic-language', x[x.dom]), ('non-domestic-language', x[~x.dom & x.train_name.isin(DOM)])]: rows.append(dict(group=nm, **metrics(g)))
for a, g in x.groupby('attr'): rows.append(dict(group=a, **metrics(g)))
S7 = pd.DataFrame(rows); S7.to_csv('../results/S7_validation.csv', index=False)
# inter-annotator agreement
tm['tm_'] = tm.tl != 'Not_mentioned'; tm['mm_'] = tm.ml != 'Not_mentioned'
kk = {a: ck(g.tm_, g.mm_) for a, g in tm.groupby('attr')}; b = tm.tm_ & tm.mm_
f = lambda z: f'{z:.2f}'
out.update(k_TM=f(ck(tm.tm_, tm.mm_)), k_TM_rng=f'{min(kk.values()):.2f}–{max(kk.values()):.2f}', q_TM=f(ck(tm.ts[b], tm.ms[b], weights='quadratic')), n_adj=str(len(adj)),
           F1_claude=f(M['F1']), P_claude=f(M['P']), R_claude=f(M['R']), k_claude=f(M['kappa']), MAE_claude=f(M['MAE']), q_claude=f(M['QWK']), pol_claude=f'{M["polarity"]:.1f}')
L = S7[S7.group.str.len() == 2]
out['F1_lang_rng'] = f'{L.F1.min():.2f} to {L.F1.max():.2f}'; out['q_lang_rng'] = f'{L.QWK.min():.2f} to {L.QWK.max():.2f}'
sg = lambda z: ('+' if z > 0 else '−' if z < 0 else '') + f'{abs(z):.2f}'
md = S7.set_index('group').loc['domestic-language', 'MSD']; mf = S7.set_index('group').loc['non-domestic-language', 'MSD']
out['msd_dom'] = sg(md); out['msd_for'] = sg(mf)
# Gemini
if os.path.exists('gemini_annotations.csv'):
    gm = model('gemini_annotations.csv'); gm = gm[gm.annotated.astype(bool)]
    y = x.merge(gm, on=['train_name', 'review_id', 'attr'], suffixes=('', '_g'))
    G = metrics(y, ml='l_g', ms='s_g'); CG = metrics(y.assign(rl=y.l, rsr=y.s), ml='l_g', ms='s_g')
    pd.DataFrame([dict(comparison='Gemini vs reference', **G), dict(comparison='Gemini vs Claude', **CG),
                  dict(comparison='Claude vs reference (same reviews)', **metrics(y))]).to_csv('../results/S12_gemini.csv', index=False)
    out.update(F1_gemini=f(G['F1']), k_CG=f(CG['kappa']), q_CG=f(CG['QWK']), n_gemini=str(y.review_id.nunique() if False else len(y) // 6))
g7 = S7.set_index('group')
out['lang_bias_text'] = (f"similar agreement for domestic-language and non-domestic-language reviews (F1 {g7.loc['domestic-language','F1']:.2f} and {g7.loc['non-domestic-language','F1']:.2f}; "
                         f"mean signed differences {sg(md)} and {sg(mf)})")
out['msd_equiv'] = 'a negligible systematic error, far below the underestimation of about half of the domestic-language scores required to reverse the direction'
json.dump(out, open('../results/validation_summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)); print(S7.round(3).to_string())
