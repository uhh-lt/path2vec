#!/usr/bin/python3
import gensim
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Loading model and semantic similarity dataset
modelfile, simfile = sys.argv[1:]

model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)

# Pre-calculating vector norms
model.init_sims(replace=True)

a = model.evaluate_word_pairs(simfile, dummy4unknown=True)

name = modelfile.replace('_embeddings_', '_')[:-7]

print(name+'\t'+str(a[1][0]))

