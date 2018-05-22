#!/usr/bin/python3
from gensim import utils
import logging
import sys
from scipy import stats

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

listfile = sys.argv[1]

similarity_gold = []
similarity_model = []

for line_no, line in enumerate(utils.smart_open(listfile)):
    line = utils.to_unicode(line)
    if line.startswith('#'):
        # May be a comment
        continue
    else:
        try:
            a, b, goldsim, sim = [word for word in line.split('\t')]
            goldsim = float(goldsim)
            sim = float(sim)
        except (ValueError, TypeError):
            logger.info('Skipping invalid line #%d in %s', line_no, listfile)
            continue

        similarity_model.append(sim)  # Similarity from the model
        similarity_gold.append(goldsim)  # Similarity from the dataset

spearman = stats.spearmanr(similarity_gold, similarity_model)
pearson = stats.pearsonr(similarity_gold, similarity_model)

logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', listfile, pearson[0], pearson[1])
logger.debug(
    'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
    listfile, spearman[0], spearman[1])

print(pearson, spearman)

