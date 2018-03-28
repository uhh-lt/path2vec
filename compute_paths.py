#!/usr/bin/python3
# coding: utf-8

import sys
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import time

ic = wordnet_ic.ic('ic-semcor.dat')

synsets = list(wn.all_synsets('n'))

print('Total synsets:', len(synsets), file=sys.stderr)

pairs = len(synsets)**2

counter = 0

start = time.time()
for el in combinations(synsets, 2):
    similarity = el[0].jcn_similarity(el[1], ic)  # Jiang-Conrath
    # similarity = el[0].lch_similarity(el[1])  # Leacock-Chodorow
    print(el[0].name() + '\t' + el[1].name() + '\t', similarity)
    counter += 1
    if counter % 100000 == 0:
        print(counter, 'out of', pairs, file=sys.stderr)

end = time.time()
print('Total time spent:', end - start)
