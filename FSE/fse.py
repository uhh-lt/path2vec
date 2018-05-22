#!/usr/bin/python3
# coding: utf-8

import sys
from gensim import utils
from nltk.corpus import wordnet as wn
from hamming_cython import hamming_sum
from itertools import product


def hamming_distance(pair, vec_dic):
    s0 = vec_dic[pair[0]]
    s1 = vec_dic[pair[1]]
    if len(s0) != len(s1):
        raise ValueError()
    distance = hamming_sum(s0, s1) + 0.00001
    return 1 / distance


if __name__ == '__main__':
    modelfile = sys.argv[1]
    pairsfile = sys.argv[2]
    model = {}
    for line in utils.smart_open(modelfile):
        line = utils.to_unicode(line)
        res = line.strip().split('\t')
        (synset, vector) = res
        model[synset] = vector
    print('Model:', len(model), file=sys.stderr)

    pairs = []
    for line in utils.smart_open(pairsfile):
        line = utils.to_unicode(line)
        if line.startswith('#'):
            # May be a comment
            continue
        res = line.strip().split('\t')
        (el0, el1, sim) = res
        synsets_el0 = wn.synsets(el0.strip(), 'n')
        synsets_el1 = wn.synsets(el1.strip(), 'n')

        if len(list(synsets_el0)) == 0 or len(list(synsets_el1)) == 0:
            print('Skipping line with words with no synsets: %s', line.strip(), file=sys.stderr)
            continue

        best_pair = None
        best_sim = 0.0
        for s_pair in product(synsets_el0, synsets_el1):
            possible_similarity = hamming_distance((s_pair[0].name(), s_pair[1].name()), model)
            if possible_similarity > best_sim:
                best_pair = s_pair
                best_sim = possible_similarity

        pairs.append((best_pair[0].name(), best_pair[1].name(), sim, str(best_sim)))

    print('Pairs ready:', len(pairs), file=sys.stderr)

    # hammings = [hamming_distance((pair[0], pair[1]), model) for pair in pairs]

    #print('Similarities calculated', file=sys.stderr)

    for i in range(len(pairs)):
        print('\t'.join(pairs[i]))
