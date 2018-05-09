#!/projects/ltg/python3/bin/python3
import gensim
import logging
import sys
from evaluate_lemmas import evaluate_synsets

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Loading model and semantic similarity dataset
modelfile = sys.argv[1]

model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)

# Pre-calculating vector norms
model.init_sims(replace=True)

if 'jcn-semcor' in modelfile:
    simfile_wordnet = 'simlex/simlex_jcn_semcor.tsv'
    simfile_humans = 'simlex/simlex_synsets/max_jcn_semcor_human.tsv'
elif 'jcn-brown' in modelfile:
    simfile_wordnet = 'simlex/simlex_jcn_brown.tsv'
    simfile_humans = 'simlex/simlex_synsets/max_jcn_brown_human.tsv'
elif 'lch' in modelfile:
    simfile_wordnet = 'simlex/simlex_lch.tsv'
    simfile_humans = 'simlex/simlex_synsets/max_lch_human.tsv'

wordnet_corr = model.evaluate_word_pairs(simfile_wordnet, dummy4unknown=True)
humans_corr = model.evaluate_word_pairs(simfile_humans, dummy4unknown=True)
dynamic_synset_score = evaluate_synsets(model, 'simlex/simlex_original.tsv', logger, dummy4unknown=True)

name = modelfile.split('/')[-1].replace('_embeddings_', '_')[:-7].replace('vsize','').replace('bsize', '')
if '_lr0.' in name:
    name = name.replace('_lr', '_')
else:
    name = name.replace('_lr', '_0.')
name = '\t'.join(name.split('_'))
output = [name, str(wordnet_corr[1][0]), str(humans_corr[1][0]), str(dynamic_synset_score[1][0])]

print('\t'.join(output))

