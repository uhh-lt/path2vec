#!/usr/bin/python3
# coding: utf-8

from gensim import models, utils
import logging
import sys
from scipy import stats
from nltk.corpus import wordnet as wn
from itertools import product


def evaluate_synsets(emb_model, pairs, our_logger, delimiter='\t', dummy4unknown=False):
    ok_vocab = [(w, emb_model.vocab[w]) for w in emb_model.index2word]
    ok_vocab = dict(ok_vocab)

    similarity_gold = []
    similarity_model = []
    oov = 0

    original_vocab = emb_model.vocab
    emb_model.vocab = ok_vocab

    for line_no, line in enumerate(utils.smart_open(pairs)):
        line = utils.to_unicode(line)
        if line.startswith('#'):
            # May be a comment
            continue
        else:
            try:
                a, b, sim = [word for word in line.split(delimiter)]
                sim = float(sim)
            except (ValueError, TypeError):
                our_logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                continue

            # Finding correct synsets
            synsets_a = wn.synsets(a.strip(), 'n')
            synsets_b = wn.synsets(b.strip(), 'n')

            if len(list(synsets_a)) == 0 or len(list(synsets_b)) == 0:
                oov += 1
                if dummy4unknown:
                    our_logger.debug('Zero similarity for line #%d with words with no synsets: %s',
                                     line_no, line.strip())
                    similarity_model.append(0.0)
                    similarity_gold.append(sim)
                    continue
                else:
                    our_logger.debug('Skipping line #%d with words with no synsets: %s',
                                     line_no, line.strip())
                    continue

            best_pair = None
            best_sim = 0.0
            for pair in product(synsets_a, synsets_b):
                possible_similarity = emb_model.similarity(pair[0].name(), pair[1].name())
                if possible_similarity > best_sim:
                    best_pair = pair
                    best_sim = possible_similarity
            our_logger.debug('Original words: %s', line.strip())
            our_logger.debug('Synsets chosen: %s with similarity %f', best_pair, best_sim)
            similarity_model.append(best_sim)  # Similarity from the model
            similarity_gold.append(sim)  # Similarity from the dataset

    emb_model.vocab = original_vocab
    spearman = stats.spearmanr(similarity_gold, similarity_model)
    pearson = stats.pearsonr(similarity_gold, similarity_model)
    if dummy4unknown:
        oov_ratio = float(oov) / len(similarity_gold) * 100
    else:
        oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

    our_logger.debug('Pearson correlation coefficient against %s: %f with p-value %f',
                     pairs, pearson[0], pearson[1])
    our_logger.debug(
        'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
        pairs, spearman[0], spearman[1])
    our_logger.debug('Pairs with unknown words: %d', oov)
    return pearson, spearman, oov_ratio


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Loading model and semantic similarity dataset
    modelfile, simfile = sys.argv[1:3]

    model = models.KeyedVectors.load_word2vec_format(modelfile, binary=False)

    # Pre-calculating vector norms
    model.init_sims(replace=True)

    scores = evaluate_synsets(model, simfile, logger, dummy4unknown=True)

    name = modelfile.replace('_embeddings_', '_')[:-7]

    print(name + '\t' + str(scores[1][0]))
