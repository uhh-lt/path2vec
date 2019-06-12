# python3

# coding: utf-8

import sys
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

testfile = sys.argv[1]
modelfiles = sys.argv[2:]

print('Model\tPearson\tSpearman')
for modelfile in modelfiles:
    model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)
    res = model.evaluate_word_pairs(testfile, case_insensitive=True, dummy4unknown=False)
    pearson = res[0]
    spearman = res[1]
    print(modelfile, '\t', '{0:.4f} with p-value {1:.5f}'.format(pearson[0], pearson[1]),
          '\t', '{0:.4f} with p-value {1:.5f}'.format(spearman[0], spearman[1]))
