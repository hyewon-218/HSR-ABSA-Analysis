"""Data-quality screening: language mismatch, platform residue, duplicates.
Usage: python 06_quality_screen.py texts.pkl quality_flags.csv
texts.pkl: train_name, review_id, new_lang (recorded language), 'yCeTE 2' (original text)."""
import sys, re, unicodedata, pandas as pd
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0
t = pd.read_pickle(sys.argv[1])
STRICT = r'Local Guide|reseñas\s*·|\d+\s*fotos|Hace (un|una|\d+) (día|días|semana|semanas|mes|meses|año|años)\S*Esta|Traducido por Google|Translated by Google|Ver original|See original|…\s?Más\b|…\s?More\b|\.\.\.\s?More$|Read more$'
def det(x):
    x = str(x)
    if len(x) < 30: return (None, 0.0)
    try:
        L = detect_langs(x)[0]; return (L.lang, L.prob)
    except Exception:
        return (None, 0.0)
d = t['yCeTE 2'].map(det)
t['det'] = [a for a, _ in d]; t['det_p'] = [b for _, b in d]
nl = t.new_lang.str[:2]; dl = t.det.fillna('').str[:2]
cjk = (nl == 'zh') & dl.isin(['ko', 'ja', 'zh']) | (nl == 'ja') & (dl == 'zh')
t['lang_mismatch'] = (t.det_p >= 0.95) & (dl != nl) & ~cjk
t['platform_residue'] = t['yCeTE 2'].astype(str).str.contains(STRICT, regex=True)
norm = t['yCeTE 2'].map(lambda s: re.sub(r'\W+', '', unicodedata.normalize('NFKC', str(s)).lower()))
t['dup'] = norm.duplicated(keep=False)
t['flag'] = t.lang_mismatch | t.platform_residue | t.dup
t[['train_name', 'review_id', 'new_lang', 'det', 'det_p', 'lang_mismatch', 'platform_residue', 'dup', 'flag']].to_csv(sys.argv[2], index=False)
print(t[['lang_mismatch', 'platform_residue', 'dup', 'flag']].sum())
