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
dhuman_scores = []


for line in sys.stdin:
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (corpus, vsize, bsize, lrate, wordnet, human, dynamic_human) = res
    if lrate != '0.005':
        continue
    if float(bsize) < 20 or float(bsize) > 400 or float(bsize) == 25:
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
    dhuman = float(dynamic_human)
    dhuman_scores.append(dhuman)


if 'jcn-semcor' in corpus:
    graph_score = 0.4874
elif 'jcn-brown' in corpus:
    graph_score = 0.4949
else:
    graph_score = 0.5134

diff_bsizes = set(batchsizes)

vectorsizes = np.array(vectorsizes)
batchsizes = np.array(batchsizes)
human_scores = np.array(human_scores)
dhuman_scores = np.array(dhuman_scores)

plt.figure()
plt.plot((0, np.max(vectorsizes)), (graph_score, graph_score), 'red', label='Pure WordNet')
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = human_scores[batchsizes == batch]
    plt.plot(x, y, linestyle='dashed', marker='o', label='Batch size '+str(int(batch)))
plt.xlabel('Vector size')
plt.ylabel('Spearman rank correlation on SimLex999')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance in semantic similarity, static synsets')
# plt.show()
plt.savefig(corpus+'_static_synsets.png', dpi=300)
plt.close()


plt.figure()
plt.plot((0, np.max(vectorsizes)), (graph_score, graph_score), 'red', label='Pure WordNet')
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = dhuman_scores[batchsizes == batch]
    plt.plot(x, y, linestyle='dashed', marker='o', label='Batch size '+str(int(batch)))
plt.xlabel('Vector size')
plt.ylabel('Spearman rank correlation on SimLex999')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance in semantic similarity, dynamic synsets')
# plt.show()
plt.savefig(corpus+'_dynamic_synsets.png', dpi=300)
plt.close()