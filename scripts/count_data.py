import json


p = '/jhcnas3/VirtualStaining/Patches/HE2PAS_256/test.json'

with open(p) as f:
    data = json.load(f)['items']

pairs = {}
for v in data:
    v = v[0].split('/')[0]
    if v not in pairs.keys():
        pairs[v] = 1
    else:
        pairs[v] += 1
        
print(pairs)