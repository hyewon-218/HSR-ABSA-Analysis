"""Re-annotate all 14,764 crawled reviews (original text, column review_text_original)
with the Gemini API, using the structured annotation prompt (SAP_prompt_v1.txt).

Run on a computer with internet access:
    pip install requests pandas openpyxl
    set GEMINI_API_KEY=...            (Windows)   /   export GEMINI_API_KEY=...   (macOS/Linux)
    python annotate_gemini_full.py --list-models            # check the exact Gemini 3 Flash model id
    python annotate_gemini_full.py --model gemini-3-flash-preview

Features
- temperature 0, JSON output, same prompt for every batch (30 reviews per batch, as in the original protocol)
- resumable: finished batches are stored in gemini_raw.jsonl and skipped on restart
- full log: request time, model version returned by the API, response id, raw response text
- after all batches: writes gemini_reannotation.csv and a QC report (missing records,
  evidence excerpts that are not found verbatim in the original text)
The API key is read from the environment and is never written to any file.
"""
import argparse, json, os, re, sys, time, datetime, unicodedata
from pathlib import Path
import pandas as pd, requests

A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']
URL = 'https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent'
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='gemini-3-flash-preview')
ap.add_argument('--input', default='reannotation_input.csv')
ap.add_argument('--prompt', default=str(Path(__file__).resolve().parents[1] / 'prompt' / 'SAP_prompt_v1.txt'))
ap.add_argument('--batch', type=int, default=30)
ap.add_argument('--raw', default='gemini_raw.jsonl')
ap.add_argument('--out', default='gemini_reannotation.csv')
ap.add_argument('--list-models', action='store_true')
ap.add_argument('--limit', type=int, default=0, help='annotate only the first N batches (pilot run)')
ap.add_argument('--workers', type=int, default=1, help='parallel requests')
a = ap.parse_args()
KEY = os.environ.get('GEMINI_API_KEY')
if not KEY:
    sys.exit('Set the environment variable GEMINI_API_KEY first.')
H = {'x-goog-api-key': KEY, 'Content-Type': 'application/json'}

if a.list_models:
    r = requests.get('https://generativelanguage.googleapis.com/v1beta/models?pageSize=200', headers=H, timeout=60)
    for m in r.json().get('models', []):
        if 'flash' in m['name']:
            print(m['name'], '|', m.get('displayName'), '|', m.get('version'))
    sys.exit()

prompt = open(a.prompt, encoding='utf-8').read()
d = pd.read_csv(a.input, encoding='utf-8-sig')
done = set()
if os.path.exists(a.raw):
    for line in open(a.raw, encoding='utf-8'):
        try:
            done.add(json.loads(line)['batch_key'])
        except Exception:
            pass

def call(text):
    body = {'systemInstruction': {'parts': [{'text': prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': text}]}],
            'generationConfig': {'temperature': 0, 'responseMimeType': 'application/json'}}
    r = requests.post(URL.format(m=a.model), headers=H, json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    txt = ''.join(p.get('text', '') for p in j['candidates'][0]['content']['parts'])
    return txt, j.get('modelVersion', a.model), j.get('responseId', '')

batches = []
for t, g in d.groupby('train_name', sort=False):
    g = g.sort_values('review_id')
    for i in range(0, len(g), a.batch):
        batches.append((f'{t}:{i // a.batch:04d}', g.iloc[i:i + a.batch]))
if a.limit:
    batches = batches[:a.limit]
print(f'{len(batches)} batches; {len(done)} already done')
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
log = open(a.raw, 'a', encoding='utf-8'); lock = threading.Lock()
def parse(txt):
    j = json.loads(txt)
    if isinstance(j, dict):
        j = next((v for v in j.values() if isinstance(v, list)), [j])
    return j
def work(bk, B):
    msg = ('Annotate the following batch of reviews. Return only the JSON array, one object per review, '
           'with train_name and review_id exactly as given. Evidence must be copied verbatim from review_text.\n\n'
           'train_name | review_id | language | review_text\n' +
           '\n'.join(f"{r.train_name} | {r.review_id} | {r.language} | {str(r.review_text_original).replace(chr(10), ' ')}" for r in B.itertuples()))
    for attempt in range(6):
        try:
            txt, ver, rid = call(msg)
            js = parse(txt)
            got = {(str(o.get('train_name')), int(o.get('review_id'))) for o in js}
            want = {(str(r.train_name), int(r.review_id)) for r in B.itertuples()}
            if not want <= got:
                raise ValueError(f'missing {len(want - got)} records')
            with lock:
                log.write(json.dumps({'batch_key': bk, 'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model_version': ver,
                                      'response_id': rid, 'records': js}, ensure_ascii=False) + '\n'); log.flush()
            return bk, True, ver
        except Exception as e:
            print(f'  {bk} attempt {attempt + 1}: {str(e)[:120]}', flush=True)
            time.sleep(15 * (attempt + 1))
    return bk, False, ''
todo = [(bk, B) for bk, B in batches if bk not in done]
with ThreadPoolExecutor(max_workers=a.workers) as ex:
    futs = [ex.submit(work, bk, B) for bk, B in todo]
    for k, f in enumerate(as_completed(futs)):
        bk, ok, ver = f.result()
        print(f'[{k + 1}/{len(todo)}] {bk} {"ok" if ok else "FAILED"} {ver}', flush=True)
log.close()

# ---- assemble + QC
norm = lambda x: re.sub(r'[^\w]+', ' ', unicodedata.normalize('NFKC', str(x)).lower()).strip()
recs = {}
meta = {}
for line in open(a.raw, encoding='utf-8'):
    j = json.loads(line)
    for o in j['records']:
        recs[(str(o.get('train_name')), int(o.get('review_id')))] = o
        meta[(str(o.get('train_name')), int(o.get('review_id')))] = (j['model_version'], j['time_utc'])
rows, qc = [], []
for r in d.itertuples():
    o = recs.get((str(r.train_name), int(r.review_id)))
    row = {'train_name': r.train_name, 'review_id': r.review_id, 'language': r.language, 'is_foreign': r.is_foreign,
           'annotated': o is not None, 'model_version': meta.get((str(r.train_name), int(r.review_id)), ('', ''))[0]}
    txt = norm(r.review_text_original)
    for x in A:
        v = (o or {}).get(x, {}) or {}
        lab = v.get('sentiment', 'Not_mentioned'); ev = v.get('evidence', '') or ''
        row[x + '_sentiment'] = lab; row[x + '_score'] = v.get('score_int', 0); row[x + '_evidence'] = ev; row[x + '_confidence'] = v.get('confidence')
        if o is not None and lab != 'Not_mentioned' and ev:
            qc.append({'train_name': r.train_name, 'review_id': r.review_id, 'attribute': x, 'evidence_verbatim': norm(ev) in txt})
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(a.out, index=False, encoding='utf-8-sig')
q = pd.DataFrame(qc)
print('annotated:', out.annotated.sum(), '/', len(out))
if len(q):
    print('evidence found verbatim in original text:', round(100 * q.evidence_verbatim.mean(), 1), '%')
    q.to_csv('gemini_qc_evidence.csv', index=False, encoding='utf-8-sig')
print('saved', a.out)
