#!/usr/bin/python3
# coding: utf-8

import sys
from scipy import stats
from compute_paths import calc_similarity
from nltk.corpus import wordnet as wn

if __name__ == '__main__':

    method = sys.argv[1]
    similarities_given = []
    similarities_wn = []

    for line in sys.stdin:
        res = line.strip().split('\t')
        (word0, word1, sim) = res
        synset0 = wn.synset(word0)
        synset1 = wn.synset(word1)
        similarities_given.append(float(sim))
        wn_sim = calc_similarity((synset0, synset1), method)
        similarities_wn.append(wn_sim)

    corr = stats.spearmanr(similarities_given, similarities_wn)

    print('Correlation:', round(corr[0], 4))
    print('P value:', corr[1])
