#!/usr/bin/python3
# coding: utf-8

from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import time

ic = wordnet_ic.ic('ic-semcor.dat')

synsets = list(wn.all_synsets('n'))


print(len(synsets))

counter = 0

start = time.time()
for el in combinations(synsets, 2):
    # similarity = el[0].jcn_similarity(el[1], ic) #  Jiang-Conrath
    similarity = el[0].lch_similarity(el[1])  # Leacock-Chodorow
    counter += 1
    if counter > 100000:
        break
end = time.time()
print(end - start)
print(counter)
