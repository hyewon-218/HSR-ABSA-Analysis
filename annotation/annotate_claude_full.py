"""Re-annotate original review texts using the Anthropic Claude API.

From the repository root:
    set ANTHROPIC_API_KEY=...  (Windows) / export ANTHROPIC_API_KEY=... (Unix)
    python annotation/annotate_claude_full.py --list-models
    python annotation/annotate_claude_full.py --model claude-sonnet-4-6 --input /private/reannotation_input.csv

Input must contain review_text_original. Raw responses/checkpoints and output
annotations are local files; API keys are read only from the environment.
"""
import argparse, json, os, re, sys, time, datetime, unicodedata
from pathlib import Path
import pandas as pd, requests

A = ['Tangibility', 'Reliability', 'Responsiveness', 'Assurance', 'Empathy', 'Comfort']
URL = 'https://api.anthropic.com/v1/messages'
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='claude-sonnet-4-6')
ap.add_argument('--input', default='reannotation_input.csv')
ap.add_argument('--prompt', default=str(Path(__file__).resolve().parents[1] / 'prompt' / 'SAP_prompt_v1.txt'))
ap.add_argument('--batch', type=int, default=30)
ap.add_argument('--raw', default='claude_raw.jsonl')
ap.add_argument('--out', default='claude_reannotation.csv')
ap.add_argument('--list-models', action='store_true')
ap.add_argument('--limit', type=int, default=0, help='annotate only the first N batches (pilot run)')
ap.add_argument('--workers', type=int, default=1, help='parallel requests')
a = ap.parse_args()
KEY = os.environ.get('ANTHROPIC_API_KEY')
if not KEY:
    sys.exit('Set the environment variable ANTHROPIC_API_KEY first.')
H = {'x-api-key': KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}

if a.list_models:
    # Send Anthropic credentials only to the Anthropic API.
    params = {'limit': 100}
    while True:
        r = requests.get('https://api.anthropic.com/v1/models', headers=H, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()
        for model in page.get('data', []):
            print(model['id'], '|', model.get('display_name', ''))
        if not page.get('has_more'):
            break
        cursor = page.get('last_id')
        if not cursor:
            raise RuntimeError('Model list pagination has no last_id')
        params['after_id'] = cursor
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

ATTR_SCHEMA = {'type': 'object', 'properties': {
    'sentiment': {'type': 'string', 'enum': ['Positive', 'Negative', 'Neutral', 'Mixed', 'Not_mentioned']},
    'score_int': {'type': 'integer', 'minimum': -2, 'maximum': 2},
    'evidence': {'type': 'string'}, 'confidence': {'type': 'integer', 'minimum': 0, 'maximum': 10}},
    'required': ['sentiment', 'score_int', 'evidence', 'confidence']}
TOOL = {'name': 'record_annotations', 'description': 'Record the annotation of every review in the batch, following the protocol.',
        'input_schema': {'type': 'object', 'properties': {'records': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'train_name': {'type': 'string'}, 'review_id': {'type': 'integer'}, 'language': {'type': 'string'},
            **{x: ATTR_SCHEMA for x in A}}, 'required': ['train_name', 'review_id'] + A}}}, 'required': ['records']}}
def call(text):
    body = {'model': a.model, 'max_tokens': 32000, 'temperature': 0, 'system': prompt,
            'tools': [TOOL], 'tool_choice': {'type': 'tool', 'name': 'record_annotations'},
            'messages': [{'role': 'user', 'content': text}]}
    r = requests.post(URL, headers=H, json=body, timeout=900)
    if r.status_code >= 400:
        raise RuntimeError(f'{r.status_code} {r.text[:200]}')
    j = r.json()
    if j.get('stop_reason') == 'max_tokens':
        raise RuntimeError('max_tokens reached')
    tu = next(c for c in j['content'] if c.get('type') == 'tool_use')
    return json.dumps(tu['input']['records'], ensure_ascii=False), j.get('model', a.model), j.get('id', '')

batches = []
for t, g in d.groupby('train_name', sort=False):
    g = g.sort_values('review_id')
    for i in range(0, len(g), a.batch):
        batches.append((f'{t}:{i // a.batch:04d}', g.iloc[i:i + a.batch]))
if a.limit:
    batches = batches[:a.limit]
SPLIT = int(os.environ.get('SPLIT_FAILED', '0'))
if SPLIT:  # re-run batches that failed at 30 reviews as sub-batches of SPLIT reviews (same model, prompt, temperature)
    sub = []
    have = set()
    for line in open(a.raw, encoding='utf-8'):
        for o in json.loads(line)['records']:
            try: have.add((str(o.get('train_name')), int(o.get('review_id'))))
            except Exception: pass
    for bk, B in batches:
        B = B[[(str(r.train_name), int(r.review_id)) not in have for r in B.itertuples()]]
        if bk in done:
            continue
        for j in range(0, len(B), SPLIT):
            sub.append((f'{bk}-n{SPLIT}-id{int(B.iloc[j].review_id)}', B.iloc[j:j + SPLIT]))
    batches = sub
print(f'{len(batches)} batches; {len(done)} already done')
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
log = open(a.raw, 'a', encoding='utf-8'); lock = threading.Lock()
def _loads(t):
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        import json_repair  # repairs unescaped quotes copied from the review text into evidence strings
        return json_repair.loads(t)
def parse(txt):
    j0 = _loads(txt)
    while isinstance(j0, str):
        j0 = _loads(j0)
    txt = json.dumps(j0)
    j = json.loads(txt, strict=False)
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
    q.to_csv('claude_qc_evidence.csv', index=False, encoding='utf-8-sig')
print('saved', a.out)
