#! /usr/bin/env python

# put the prediction file in order (parallel evaluation can reorder it)

# python evaluate_v1.0.py <(~/arf/musique_clean/results/reorder_preds_like.py "$1" "$2") "$2"
# ./reorder_preds_like.py qwen32.val.baseline.preds ~/musique/data/musique_ans_v1.0_dev.jsonl

import hashlib
import json
import sys

preds = sys.argv[1]
golds = sys.argv[2]

predlines = {}
predobjs = []
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

for goldline, predline in sorted(reorders.items()):
    print(json.dumps(predobjs[predline]))
