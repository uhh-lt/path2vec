#!/projects/ltg/python3/bin/python3
# coding: utf-8

import sys

threshold = float(sys.argv[1])
total = 0
pruned = 0

minim = threshold
maxim = 2.6

for line in sys.stdin:
    res = line.strip().split('\t')
    if len(res) != 3:
        print(line.strip(), file=sys.stderr)
        continue
    (synset0, synset1, similarity) = res
    similarity = float(similarity)
    total += 1
    if similarity > threshold:
        new_sim = (similarity - minim) / (maxim - minim)
        if new_sim > 1.0:
            new_sim = 1.0
        print('\t'.join([synset0, synset1, str(new_sim)]))
    else:
        pruned += 1

print('Threshold was:', threshold, file=sys.stderr)
print('Pruned', pruned, 'out of  total', total, file=sys.stderr)
