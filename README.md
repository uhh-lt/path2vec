# vector-paths
Learning to represent shortest paths with graph embeddings

# Models evaluation

`python3 evaluation.py MODELFILE SIMFILE`

MODELFILE is the file with synset vectors in word2vec text format.

SIMFILE is one of semantic similarity datasets in https://github.com/uhh-lt/shortpath2vec/blob/master/simlex/simlex_synsets/

For example:

`python3 evaluation.py jcn_brown_thresh01_embeddings_vsize16_bsize20_lr004.vec.gz simlex_synsets/max_jcn_brown_human.tsv`

`jcn_brown_thresh01_vsize16_bsize20_lr004     0.2312464827703736`

The resuting score is the Spearman rank correlation between model-produces similarities and human judgments.
