# path2vec
Learning to represent shortest paths and other graph-based measures of node similarities with graph embeddings. This repository contains code related to this paper:  

Andrey Kutuzov, Alexander Panchenko, Sarah Kohail, Mohammad Dorgham, Oleksiy Oliynyk, Chris Biemann (2018) [Learning Graph Embeddings from WordNet-based Similarity Measures](https://arxiv.org/abs//1808.05611).

We present a new approach for learning graph embeddings, that relies on structural measures of node similarities for generation of training data. The model learns node embeddings that are able to approximate a given measure, such as the shortest path distance or any other. Evaluations of the proposed model on semantic similarity and word sense disambiguation tasks (using WordNet as the source of gold similarities) show that our method yields state-of-the-art results, but also is capable in certain cases to yield even better performance than the input similarity measure. The model is computationally efficient, orders of magnitude faster than the direct computation of graph distances.

# Models evaluation

`python3 evaluation.py MODELFILE SIMFILE`

MODELFILE is the file with synset vectors in word2vec text format.

SIMFILE is one of semantic similarity datasets in https://github.com/uhh-lt/shortpath2vec/blob/master/simlex/simlex_synsets/

For example:

`python3 evaluation.py jcn_semcor_thresh01_embeddings_vsize200_bsize100_lr001.vec.gz simlex/simlex_synsets/max_jcn_semcor_human.tsv`

`jcn_semcor_thresh01_vsize200_bsize100_lr001     0.46397722955881243     0.4503616841776444`

The resuting score (0.464 in the example) is the Spearman rank correlation between model-produced similarities and human judgments.
The second score (0.450 in the example) is always calculated on the original Simlex with dynamically selected synsets
(see below for details).

# Evaluation with dynamic synset selection

One can also evaluate using dynamic synset selection on the original SimLex test set
(https://github.com/uhh-lt/shortpath2vec/blob/master/simlex/simlex_original.tsv )

'Dynamic synset selection' here means that the test set contains lemmas, not synsets.
From all possible WordNet synsets for words A and B in each test set pair, we choose the synset combination which yields
maximum similarity in the model under evaluation. For example, for the words `weekend` and `week` we choose the synsets
`weekend.n.01` and `workweek.n.01`, etc.

To evaluate the model this way, use the `evaluate_lemmas.py` script:

`python3 evaluate_lemmas.py MODELFILE simlex/simlex_original.tsv`
