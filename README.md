# path2vec
This repository contains code related to this paper:  

Andrey Kutuzov, Mohammad Dorgham, Oleksiy Oliynyk, Chris Biemann, Alexander Panchenko (2019)

[Making Fast Graph-based Algorithms with Graph Metric Embeddings](https://aclweb.org/anthology/papers/P/P19/P19-1325/)

_Path2vec_ is a new approach for learning graph embeddings that relies on structural measures of pairwise node similarities. 
The model learns representations for nodes in a dense space that approximate a given user-defined graph distance measure, such as e.g. the shortest path distance or distance measures that take information beyond the graph structure into account. 
Evaluation of the model on semantic similarity and word sense disambiguation tasks, using various WordNet-based similarity measures, show that our approach yields competitive results, outperforming strong graph embedding baselines. 
The model is computationally efficient, being orders of magnitude faster than the direct computation of graph-based distances.

# Pre-trained models and datasets
You can download pre-trained dense vector representations of WordNet synsets approximating several different graph distance metrics:
- [Jiang-Conrath (SemCor)](http://ltdata1.informatik.uni-hamburg.de/path2vec/embeddings/jcn-semcor_embeddings.vec.gz)
- [Leacock-Chodorow](http://ltdata1.informatik.uni-hamburg.de/path2vec/embeddings/lch_embeddings.vec.gz)
- [Shortest path](http://ltdata1.informatik.uni-hamburg.de/path2vec/embeddings/shp_embeddings.vec.gz)
- [Wu-Palmer](http://ltdata1.informatik.uni-hamburg.de/path2vec/embeddings/wup_embeddings.vec.gz)

Prepared training datasets are also available:
- [datasets](https://ltnas1.informatik.uni-hamburg.de:8081/owncloud/index.php/s/lhcJQNxaGBLjL8o?path=%2Fdatasets)

# Models training
Train your own graph embedding model with:

`python3 embeddings.py --input_file TRAINING_DATASET --vocab_file synsets_vocab.json.gz --use_neighbors`

Run `python3 embeddings.py -h` for help on tunable hyperparameters.

# Models evaluation

`python3 evaluation.py MODELFILE SIMFILE0 SIMFILE1`

`MODELFILE` is the file with synset vectors in word2vec text format.

`SIMFILE` is one of [semantic similarity datasets](https://github.com/uhh-lt/path2vec/tree/master/simlex/). 
It is expected that `SIMFILE0` will contain Wordnet similarities, while `SIMFILE1` will contain SimLex999 similarities, 
and that they correspond to the graph distance metrics on which the model was trained. 
The model will be tested on both of these test sets, and additionally on the raw SimLex999 (dynamically assigning synsets to lemmas).

For example, to evaluate on the shortest path metrics (`shp`):

`python3 evaluation.py shp.vec.gz simlex/simlex_shp.tsv simlex/simlex_synsets/max_shp_human.tsv`

`Model  Wordnet Static  Dynamic`

`shp 0.9473  0.5121  0.5551`

The resulting score 0.9473 is the Spearman rank correlation between model-produced similarities and WordNet similarities (using `SIMFILE0`). 
The second score 0.5121 is calculated on `SIMFILE1` (human judgments). 
The 3rd score (0.5551 in the example) is always calculated on the original Simlex with dynamically selected synsets (see below for details).

# Evaluation with dynamic synset selection

One can also evaluate using dynamic synset selection on the [original SimLex test set](https://github.com/uhh-lt/shortpath2vec/blob/master/simlex/simlex_original.tsv).

'Dynamic synset selection' here means that the test set contains lemmas, not synsets.
From all possible WordNet synsets for words A and B in each test set pair, we choose the synset combination which yields maximum similarity in the model under evaluation. 
For example, for the words `weekend` and `week` we choose the synsets `weekend.n.01` and `workweek.n.01`, etc.

To evaluate the model this way, use the `evaluate_lemmas.py` script:

`python3 evaluate_lemmas.py MODELFILE simlex/simlex_original.tsv`

# BibTex
```
@inproceedings{kutuzov-etal-2019-making,
    title = "Making Fast Graph-based Algorithms with Graph Metric Embeddings",
    author = "Kutuzov, Andrey  and
      Dorgham, Mohammad  and
      Oliynyk, Oleksiy  and
      Biemann, Chris  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1325",
    pages = "3349--3355",
    abstract = "Graph measures, such as node distances, are inefficient to compute. We explore dense vector representations as an effective way to approximate the same information. We introduce a simple yet efficient and effective approach for learning graph embeddings. Instead of directly operating on the graph structure, our method takes structural measures of pairwise node similarities into account and learns dense node representations reflecting user-defined graph distance measures, such as e.g. the shortest path distance or distance measures that take information beyond the graph structure into account. We demonstrate a speed-up of several orders of magnitude when predicting word similarity by vector operations on our embeddings as opposed to directly computing the respective path-based measures, while outperforming various other graph embeddings on semantic similarity and word sense disambiguation tasks.",
}
```
