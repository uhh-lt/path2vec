#!/usr/bin/python3
# coding: utf-8

import sys
import gensim
import logging
from itertools import combinations


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

modelfile = sys.argv[1]

if modelfile.endswith('.bin.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=True)
else:
    model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)


word0 = 'measure.n.02'
word1 = 'fundamental_quantity.n.01'
word2 = 'person.n.01'
word3 = 'lover.n.03'


for pair in combinations([word0, word1, word2, word3], 2):
    print(pair, model.similarity(pair[0], pair[1]))
