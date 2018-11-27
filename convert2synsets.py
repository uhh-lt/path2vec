#!/usr/bin/python3
# coding: utf-8

import sys
from itertools import product
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from compute_paths import calc_similarity

if __name__ == '__main__':
    method = sys.argv[1]  # jcn or lch or path or wup
    corpus = sys.argv[2]  # semcor or brown

    preserve = False  # Preserve SimLex similarities?

    if len(sys.argv) > 3:
        preserve = True

    maxval = 1000.0  # This value will be assigned to extremely high-similarity pairs (like 1e+300)

    if corpus != 'None':
        ic = wordnet_ic.ic('ic-%s.dat' % corpus)
    else:
        ic = None

    for line in sys.stdin:
        if line.strip().startswith('#'):
            continue
        res = line.strip().split('\t')
        (word0, word1, simlex_sim) = res
        simlex_sim = float(simlex_sim)
        synsets0 = reversed(wn.synsets(word0.strip(), 'n'))
        synsets1 = reversed(wn.synsets(word1.strip(), 'n'))
        best_pair = None
        best_sim = 0.0
        for pair in product(synsets0, synsets1):
            if pair[0] == pair[1]:
                continue
            wordnet_sim = calc_similarity(pair, method, infcont=ic)
            if wordnet_sim > best_sim:
                best_pair = pair
                best_sim = wordnet_sim
        if not best_pair:
            print('Weird data:', line, file=sys.stderr)
            if preserve:
                print('\t'.join([s.name() for s in pair]) + '\t' + str(simlex_sim))
            else:
                print('\t'.join([s.name() for s in pair]) + '\t' + str(maxval))
            continue
        if preserve:
            best_sim = simlex_sim
        if best_sim > 1000:
            print('Clipped similarity to %f' % maxval, best_pair, best_sim, file=sys.stderr)
            best_sim = maxval
        if best_sim < 0.0001:
            print('Clipped similarity to 0.0', best_pair, best_sim, file=sys.stderr)
            best_sim = 0.0
        print('\t'.join([s.name() for s in best_pair]) + '\t' + str(best_sim))
