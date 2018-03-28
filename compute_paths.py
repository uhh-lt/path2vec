#!/usr/bin/python3
# coding: utf-8

import sys
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import time
import multiprocessing
from math import factorial
from multiprocessing import Pool


def calc_similarity(pair):
    syns0 = wn.synset(pair[0])
    syns1 = wn.synset(pair[1])
    if method == 'jcn':
        similarity = syns0.jcn_similarity(syns1, ic)  # Jiang-Conrath
    elif method == 'lch':
        similarity = syns0.lch_similarity(syns1)  # Leacock-Chodorow
    else:
        return None
    print(pair[0] + '\t' + pair[1] + '\t', similarity)
    return similarity


if __name__ == '__main__':
    method = sys.argv[1]

    cores = multiprocessing.cpu_count()-1
    ic = wordnet_ic.ic('ic-semcor.dat')
    synsets = list(wn.all_synsets('n'))
    synsets = [s.name() for s in synsets]
    print('Total synsets:', len(synsets), file=sys.stderr)
    print('Method:', method, file=sys.stderr)

    print('Calculating total number of pairs...', file=sys.stderr)
    pairs = factorial(len(synsets)) // factorial(2) // factorial(len(synsets) - 2)
    print(pairs, file=sys.stderr)

    synset_pairs = combinations(synsets, 2)

    counter = 0
    start = time.time()

    with Pool(cores) as p:
        for i in p.imap_unordered(calc_similarity, synset_pairs, chunksize=100):
            counter += 1
            if counter % 100000 == 0:
                print(counter, 'out of', pairs, file=sys.stderr)

    end = time.time()
    print('Total time spent:', end - start, file=sys.stderr)
