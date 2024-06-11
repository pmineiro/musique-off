#! /usr/bin/env python

# put the prediction file in order (parallel evaluation can reorder it)
# futhermore enforce the constraint: "for each pair, exactly one is answerable and exactly one is not"

# python evaluate_v1.0.py <(~/arf/musique_clean/results/reorder_preds_like_full_pair.py "$1" "$2") "$2"
# ./reorder_preds_like_full_pair.py qwen32.val.baseline.preds ~/musique/data/musique_full_v1.0_dev.jsonl

from collections import defaultdict
import hashlib
import json
import sys

preds = sys.argv[1]
golds = sys.argv[2]

predlines = {}
predobjs = []
objectpairs = defaultdict(list)
have_pardigest = False

with open(preds, 'r') as file:
    for n, line in enumerate(file):
        obj = json.loads(line)
        have_pardigest = have_pardigest or 'pardigest' in obj
        pardigest = obj.pop('pardigest', None)
        key = (obj['id'], pardigest)
        assert key not in predlines, f"duplicate {key} in preds"
        predlines[key] = n
        predobjs.append(obj)
        objectpairs[obj['id']].append(obj)

adjusted = 0
notpaired = 0
norawscore = 0
for key, n in predlines.items():
    objid = key[0]
    pair = objectpairs[objid]
    if len(pair) < 2:
        notpaired += 1
    elif pair[0]['predicted_answerable'] == pair[1]['predicted_answerable']:
        adjusted += 1
        if any('suff_raw_score' not in p for p in pair):
            norawscore += 1
            best, worst = sorted(pair, key=lambda v:-len(v['predicted_support_idxs']), reverse=True)
            best['predicted_answerable'] = True
            worst['predicted_answerable'] = False
        else:
            best, worst = sorted(pair, key=lambda v:v['suff_raw_score'], reverse=True)
            best['predicted_answerable'] = True
            worst['predicted_answerable'] = False
            del best['suff_raw_score']
            del worst['suff_raw_score']

print(f'adjusted = {adjusted}*2 = {adjusted*2}', file=sys.stderr)
print(f'notpaired = {notpaired}', file=sys.stderr)
print(f'norawscore = {norawscore}', file=sys.stderr)

reorders = {}
seen = set()
with open(golds, 'r') as file:
    for n, line in enumerate(file):
        obj = json.loads(line)
        partext = '\n\n'.join(p['paragraph_text'] for p in obj['paragraphs'])
        pardigest = hashlib.sha256(partext.encode('utf-8')).hexdigest()
        key = (obj['id'], pardigest) if have_pardigest else (obj['id'], None)
        assert key not in seen, f"duplicate {key} in golds"
        seen.add(key)
        reorders[n] = predlines.get(key, -1)

missing = sum(int(r < 0) for r in reorders.values())
print(f'missing = {missing}', file=sys.stderr)
matching = sum(int(r >= 0) for r in reorders.values())
print(f'matching = {matching}', file=sys.stderr)
inorder = sum(n == v for n, v in reorders.items() if v >= 0)
print(f'inorder = {inorder}', file=sys.stderr)

assert missing == 0
assert notpaired == 0

for goldline, predline in sorted(reorders.items()):
    print(json.dumps(predobjs[predline]))
