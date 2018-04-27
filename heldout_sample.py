#!/usr/bin/python3
# coding: utf-8

import sys
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from convert2synsets import calc_similarity


if __name__ == '__main__':
    method = sys.argv[1]  # jcn or lch
    corpus = sys.argv[2]  # semcor or brown

    maxval = 1000.0  # This value will be assigned to extremely high-similarity pairs (like 1e+300)

    ic = wordnet_ic.ic('ic-%s.dat' % corpus)

    words = set()

    for line in sys.stdin:
        if line.strip().startswith('#'):
            continue
        res = line.strip().split('\t')
        (word0, word1, sim) = res
        words.add(word0.strip())
        words.add(word1.strip())

    synsets = set()
    for w in words:
        w_synsets = wn.synsets(w, 'n')
        for s in w_synsets:
            synsets.add(s)

    print('%d synsets produced from %d words' % (len(synsets), len(words)), file=sys.stderr)

    all_synsets = list(wn.all_synsets('n'))
    for synset in synsets:
        print('Calculating similarities for', synset, file=sys.stderr)
        for s in all_synsets:
            if s != synset:
                pair = (synset, s)
                similarity = calc_similarity(pair, method, ic)
                if similarity > 1000:
                    print('Clipped similarity to %f' % maxval, pair, similarity, file=sys.stderr)
                    similarity = maxval
                if similarity < 0.0001:
                    print('Clipped similarity to 0.0', pair, similarity, file=sys.stderr)
                    similarity = 0.0
                print('\t'.join([s.name() for s in pair]) + '\t' + str(similarity))
