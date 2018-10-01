#!/usr/bin/python3
import gensim
import logging
import sys
from evaluate_lemmas import evaluate_synsets

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading model and semantic similarity dataset
modelfile, simfile = sys.argv[1:]

model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)

# Pre-calculating vector norms
model.init_sims(replace=True)

static_synset_score = model.evaluate_word_pairs(simfile, dummy4unknown=True)
dynamic_synset_score = evaluate_synsets(model, 'simlex/simlex_original.tsv', logger, dummy4unknown=True)

name = modelfile.replace('_embeddings_', '_')[:-7]

print('Model\tStatic\tDynamic')
print(name + '\t' + str(static_synset_score[1][0]) + '\t' + str(dynamic_synset_score[1][0]))
