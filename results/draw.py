#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

vectorsizes = []
batchsizes = []
lrates = []
wordnet_scores = []
human_scores = []


for line in sys.stdin:
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (corpus, vsize, bsize, lrate, wordnet, human) = res
    if lrate != '0.001':
        continue
    lrates.append(lrate)
    vsize = float(vsize)
    vectorsizes.append(vsize)
    bsize = float(bsize)
    batchsizes.append(bsize)
    wordnet = float(wordnet)
    wordnet_scores.append(wordnet)
    human = float(human)
    human_scores.append(human)


if corpus == 'jcn-semcor':
    graph_score = 0.4874
elif corpus == 'jcn-brown':
    graph_score = 0.4949
else:
    graph_score = 0.5134

diff_bsizes = set(batchsizes)

vectorsizes = np.array(vectorsizes)
batchsizes = np.array(batchsizes)
human_scores = np.array(human_scores)

plt.figure()
plt.plot((0, 300), (graph_score, graph_score), 'red', label='Pure WordNet')
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = human_scores[batchsizes == batch]
    plt.plot(x, y, linestyle='dashed', marker='o', label='Batch size '+str(int(batch)))
plt.xlabel('Vector size')
plt.ylabel('Spearman rank correlation on SimLex')
plt.legend(loc='best')
plt.grid(True)
plt.title(corpus)
#plt.show()
plt.savefig(corpus+'.png', dpi=300)
plt.close()
