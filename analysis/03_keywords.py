"""Weighted log-odds ratio (Monroe et al., 2008) between non-domestic-language (F) and domestic-language (D) reviews,
pooled within clusters, reproduced from aggregated word counts (review texts are not redistributed).
Usage (from the repository root): python analysis/03_keywords.py results"""
import sys, json
import numpy as np, pandas as pd
from collections import Counter
OUT = sys.argv[1]; CLU = json.load(open(f'{OUT}/results.json'))['clusters']
STOP = set([
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
STOP |= {'hongqiao', 'xian', 'guangzhou', 'shenzhen', 'hangzhou', 'nanjing', 'suzhou', 'hiroshima', 'kyushu', 'fukuoka', 'kobe', 'yokohama', 'daegu', 'gyeongju',
        'kaohsiung', 'taichung', 'tainan', 'zuoying', 'naples', 'napoli', 'bologna', 'lyon', 'marseille', 'nice', 'bordeaux', 'strasbourg', 'frankfurt', 'hamburg',
        'cologne', 'stuttgart', 'nuremberg', 'seville', 'valencia', 'malaga', 'cordoba', 'brussels', 'amsterdam', 'renfe', 'avlo', 'eurostar', 'inoui', 'trenitalia', 'kodama', 'nozomi', 'hikari', 'tokaido', 'shin', 'tgvs', 'korail', 'frecce', 'freccia', 'deutsche', 'thsrc', 'taoyuan', 'hsinchu'} | {'doesn', 'didn', 'isn', 'wasn', 'nach', 'gare', 'arrow', 'shizuoka', 'north', 'thello'}
KW = {'Tangibility': ['seat', 'seats', 'clean', 'cleanliness', 'dirty', 'bathroom', 'bathrooms', 'toilet', 'toilets', 'food', 'wifi', 'space', 'luggage', 'carriage', 'carriages', 'window', 'seating'],
      'Reliability': ['delay', 'delays', 'delayed', 'late', 'cancel', 'cancelled', 'canceled', 'strike', 'punctual', 'schedule', 'efficient', 'missed', 'unpunctual'],
      'Responsiveness': ['compensation', 'refund', 'informed', 'assistance', 'response', 'help', 'problem', 'problems', 'queue'],
      'Assurance': ['staff', 'rude', 'helpful', 'english', 'language', 'announcement', 'announcements', 'speak', 'understand', 'polite', 'friendly'],
      'Empathy': ['expensive', 'cheap', 'cheaper', 'price', 'prices', 'cost', 'affordable', 'overpriced', 'money', 'children', 'family'],
      'Comfort': ['smooth', 'quiet', 'cold', 'conditioning', 'temperature', 'noise', 'noisy', 'crowded', 'standing', 'smell', 'odor', 'comfortable', 'uncomfortable']}
ALPHA = 0.1
wc = pd.read_csv('data/word_counts_by_system_group.csv', keep_default_na=False); wc = wc[~wc.word.isin(STOP)]
wc['cluster'] = wc.system.map(CLU)
def logodds(cf, cd):
    V = set(cf) | set(cd); nf, nd = sum(cf.values()), sum(cd.values()); k = len(V); a0 = k * ALPHA; z = {}
    for w in V:
        yf, yd = cf[w], cd[w]
        dlt = np.log((yf + ALPHA) / (nf + a0 - yf - ALPHA)) - np.log((yd + ALPHA) / (nd + a0 - yd - ALPHA))
        z[w] = dlt / np.sqrt(1 / (yf + ALPHA) + 1 / (yd + ALPHA))
    return z
rows, attr = [], []
for translated_only in [False, True]:
    for c in [1, 2, 3]:
        s = wc[wc.cluster == c]
        if translated_only: s = s[(s.group == 'dom') | ~s.english_original]
        cf = Counter(s[s.group == 'foreign'].groupby('word')['count'].sum().to_dict()); cd = Counter(s[s.group == 'dom'].groupby('word')['count'].sum().to_dict())
        z = logodds(cf, cd); srt = sorted(z.items(), key=lambda x: (-x[1], x[0]))
        for rk, (w, v) in enumerate(srt[:30], 1): rows.append(dict(translated_only=translated_only, cluster=c, group='F', rank=rk, word=w, z=v, attr=next((a for a, l in KW.items() if w in l), '')))
        for rk, (w, v) in enumerate(srt[::-1][:30], 1): rows.append(dict(translated_only=translated_only, cluster=c, group='D', rank=rk, word=w, z=v, attr=next((a for a, l in KW.items() if w in l), '')))
        if not translated_only:
            for a, l in KW.items():
                v = [z[w] for w in l if w in z]; attr.append(dict(cluster=c, attr=a, Z_a=float(np.mean(v)) if v else np.nan, n_kw=len(v)))
        else:
            for w in ['english', 'clean']:
                rk = [i for i, (ww, _) in enumerate(srt, 1) if ww == w]
                rows.append(dict(translated_only=True, cluster=c, group='rank_of', rank=rk[0] if rk else None, word=w, z=z.get(w, np.nan), attr=''))
pd.DataFrame(rows).to_csv(f'{OUT}/keywords.csv', index=False); at = pd.DataFrame(attr); at.to_csv(f'{OUT}/table7_Za.csv', index=False)
print(at.pivot(index='attr', columns='cluster', values='Z_a').round(3))
