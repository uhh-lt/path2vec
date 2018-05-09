#!/projects/ltg/python3/bin/python3
# coding: utf-8

import sys
import numpy as np

similarities_dict = {}

no_neighbors = int(sys.argv[1])

for line in sys.stdin:
    res = line.strip().split('\t')
    if len(res) != 3:
        print(line.strip(), file=sys.stderr)
        continue
    (synset0, synset1, similarity) = res
    similarity = float(similarity)
    if not synset0 in similarities_dict:
        similarities_dict[synset0] = {}
    similarities_dict[synset0][synset1] = similarity

print('We have dictionary of length', len(similarities_dict), file=sys.stderr)

for synset in similarities_dict:
    nearest = sorted(similarities_dict[synset], key=similarities_dict[synset].get, reverse=True)[:no_neighbors]
    for neighbor in nearest:
        print(synset+'\t'+neighbor+'\t'+str(similarities_dict[synset][neighbor]))

