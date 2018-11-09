#!/usr/bin/python3
import gensim
import logging
import sys
from evaluate_lemmas import evaluate_synsets

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading model and semantic similarity dataset
modelfile, wordnet_scores, static_scores = sys.argv[1:]

model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)

# Pre-calculating vector norms
model.init_sims(replace=True)

wordnet_synset_score = model.evaluate_word_pairs(wordnet_scores, dummy4unknown=True)
static_synset_score = model.evaluate_word_pairs(static_scores, dummy4unknown=True)
dynamic_synset_score = evaluate_synsets(model, 'simlex/simlex_original.tsv', logger, dummy4unknown=True)

name = modelfile.replace('_embeddings_', '_')[:-7]

print('Model\tWordnet\tStatic\tDynamic')
print(name + '\t' + str(round(wordnet_synset_score[1][0], 4)) + '\t' + str(round(static_synset_score[1][0], 4)) + '\t' + str(round(dynamic_synset_score[1][0], 4)))
