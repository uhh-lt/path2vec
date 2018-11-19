# path2vec
Learning to represent shortest paths and other graph-based measures of node similarities with graph embeddings. This repository contains code related to this paper:  

Andrey Kutuzov, Alexander Panchenko, Sarah Kohail, Mohammad Dorgham, Oleksiy Oliynyk, Chris Biemann (2018) [Learning Graph Embeddings from WordNet-based Similarity Measures](https://arxiv.org/abs//1808.05611).

We present a new approach for learning graph embeddings, that relies on structural measures of node similarities for generation of training data. The model learns node embeddings that are able to approximate a given measure, such as the shortest path distance or any other. Evaluations of the proposed model on semantic similarity and word sense disambiguation tasks (using WordNet as the source of gold similarities) show that our method yields state-of-the-art results, but also is capable in certain cases to yield even better performance than the input similarity measure. The model is computationally efficient, orders of magnitude faster than the direct computation of graph distances.

# Models evaluation

`python3 evaluation.py MODELFILE SIMFILE0 SIMFILE1`

MODELFILE is the file with synset vectors in word2vec text format.

SIMFILE is one of semantic similarity datasets in https://github.com/uhh-lt/path2vec/tree/master/simlex or https://github.com/uhh-lt/path2vec/tree/master/simlex/simlex_synsets. It is expected that SIMFILE0 will be from the first directory (Wordnet similarities), and SIMFILE1 will be from the second one (SimLex999 similarities), and that they correspond to the graph distance metrics on which the model was trained. The model will be tested on both of these test sets, and additionally on the raw SimLex999 (dynamically assigning synsets to lemmas).

For example, to evaluate on the shortest path metrics (`shp'):

`python3 evaluation.py shp_thresh01-near50_embeddings_vsize300_bsize100_lr001_nn-True3_reg-True_shp.vec.gz simlex/simlex_shp.tsv simlex/simlex_synsets/max_shp_human.tsv`

`Model  Wordnet Static  Dynamic`

`shp_thresh01-near50_vsize300_bsize100_lr001_nn-True3_reg-True_shp 0.9473  0.5121  0.5551`

The resuting score 0.9473 is the Spearman rank correlation between model-produced similarities and WordNet similarities (using SIMFILE0). The second score 0.5121 is calculated on SIMFILE1 (human judgments). The 3rd score (0.5551 in the example) is always calculated on the original Simlex with dynamically selected synsets (see below for details).

# Evaluation with dynamic synset selection

One can also evaluate using dynamic synset selection on the original SimLex test set
(https://github.com/uhh-lt/shortpath2vec/blob/master/simlex/simlex_original.tsv )

'Dynamic synset selection' here means that the test set contains lemmas, not synsets.
From all possible WordNet synsets for words A and B in each test set pair, we choose the synset combination which yields
maximum similarity in the model under evaluation. For example, for the words `weekend` and `week` we choose the synsets
`weekend.n.01` and `workweek.n.01`, etc.

To evaluate the model this way, use the `evaluate_lemmas.py` script:

`python3 evaluate_lemmas.py MODELFILE simlex/simlex_original.tsv`
