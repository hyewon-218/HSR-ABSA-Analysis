"""Relation between overall star rating and attribute-level sentiment (model annotations)."""
import pandas as pd, numpy as np, json, warnings, os
os.makedirs('results/R2_star', exist_ok=True)
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['Tangibility','Reliability','Responsiveness','Assurance','Empathy','Comfort']
DOM={'KTX':'ko','Shinkansen':'ja','THSR':'zh','ICE':'de','Frecciarossa':'it','Italo':'it','TGV':'fr','RENFE':'es','CRH':'zh'}
d=pd.read_csv('data/claude_sonnet46_annotations.csv',encoding='utf-8-sig').merge(pd.read_csv('data/review_metadata.csv'),on=['train_name','review_id'])
for a in A:
    l=d[a+'_sentiment']; s=pd.to_numeric(d[a+'_score'],errors='coerce').fillna(0).clip(-2,2); s[l.isin(['Mixed','Neutral'])]=0
    d[a+'_m']=(l!='Not_mentioned').astype(float); d[a+'_s']=np.where(l!='Not_mentioned',s,0.0); d[a+'_pol']=np.where(l=='Not_mentioned',np.nan,np.sign(s))
d=d[d.star.notna()].copy(); d['dom']=[DOM.get(t)==l for t,l in zip(d.train_name,d.language)]
O={}
# 1. discordance between overall rating and attribute-level polarity
pos=(d[[a+'_pol' for a in A]]>0).sum(1); neg=(d[[a+'_pol' for a in A]]<0).sum(1); men=d[[a+'_m' for a in A]].sum(1)
O['n']=int(len(d)); O['both_pos_and_neg_attr']=round(float(((pos>0)&(neg>0)).mean()*100),1)
O['high_star_with_negative_attr']=round(float((neg[d.star>=4]>0).mean()*100),1); O['low_star_with_positive_attr']=round(float((pos[d.star<=2]>0).mean()*100),1)
O['n_high']=int((d.star>=4).sum()); O['n_low']=int((d.star<=2).sum())
O['rho_attr_star']={a:round(float(spearmanr(d.loc[d[a+'_m']==1,a+'_s'],d.loc[d[a+'_m']==1,'star'])[0]),2) for a in A}
# 2. attribute weights: cumulative logit of star on attribute scores (0 if not mentioned) + mention indicators + system FE
def fit(sub):
    X=pd.concat([sub[[a+'_s' for a in A]],sub[[a+'_m' for a in A]],pd.get_dummies(sub.train_name,drop_first=True).astype(float)],axis=1); X=X.loc[:,X.std()>0]
    r=OrderedModel(sub.star.astype(int).values,X.values,distr='logit').fit(method='bfgs',disp=False,maxiter=5000)
    cols=list(X.columns); return {a:(float(r.params[cols.index(a+'_s')]),float(r.bse[cols.index(a+'_s')])) for a in A}, r
W,_=fit(d); O['beta_all']={a:[round(b,3),round(se,3)] for a,(b,se) in W.items()}
# 3. do weights differ by language group? separate fits and z-test of difference
sub={'dom':d[d.dom],'nondom':d[(~d.dom)&d.train_name.isin(DOM)]}
WD,_=fit(sub['dom']); WN,_=fit(sub['nondom'])
rows=[]
for a in A:
    (b1,s1),(b2,s2)=WD[a],WN[a]; z=(b2-b1)/np.sqrt(s1**2+s2**2)
    from scipy.stats import norm; p=min(1,6*2*norm.sf(abs(z)))
    rows.append(dict(attr=a,beta_dom=round(b1,3),se_dom=round(s1,3),beta_nondom=round(b2,3),se_nondom=round(s2,3),z_diff=round(z,2),p_bonf=round(p,4)))
O['weights_by_group']=rows; O['n_dom']=int(len(sub['dom'])); O['n_nondom']=int(len(sub['nondom']))
# 4. relative importance: share of explained variance (dominance-style, Shapley over 6 attribute blocks, OLS on star with system FE)
from itertools import combinations
import numpy.linalg as la
FE=pd.get_dummies(d.train_name,drop_first=True).astype(float).values; y=d.star.values.astype(float)
def r2(S):
    X=np.column_stack([np.ones(len(d)),FE]+[d[[a+'_s',a+'_m']].values for a in S]); b=la.lstsq(X,y,rcond=None)[0]; e=y-X@b; return 1-e.var()/y.var()
base=r2([]); cache={}
def R(S):
    k=tuple(sorted(S)); 
    if k not in cache: cache[k]=r2(list(k))
    return cache[k]
from math import factorial
sh={}
for a in A:
    others=[x for x in A if x!=a]; v=0
    for k in range(6):
        for S in combinations(others,k): v+=factorial(k)*factorial(5-k)/factorial(6)*(R(S+(a,))-R(S))
    sh[a]=v
tot=R(tuple(A))-base; O['R2_FE_only']=round(base,3); O['R2_full']=round(R(tuple(A)),3); O['shapley_share_pct']={a:round(100*v/tot,1) for a,v in sh.items()}
json.dump(O,open('results/R2_star/star_attr.json','w'),indent=1,ensure_ascii=False); print(json.dumps(O,indent=1,ensure_ascii=False))
