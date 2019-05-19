#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') 
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
    if float(bsize) < 100 or float(bsize) > 366 or float(bsize) == 265:
        continue
    lrates.append(lrate)
    vsize = float(vsize)
    if vsize < 100:
        continue
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
plt.plot((50, np.max(vectorsizes)), (graph_score, graph_score), 'red', label='Pure WordNet')
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = human_scores[batchsizes == batch]
    if int(batch) == 166:
        label = 'Deepwalk'
    elif int(batch) == 366:
        label = 'node2vec'
    elif int(batch) == 266:
        label = 'TransR'
    elif int(batch) == 265:
        label = 'TransD'
    else:
        #label = 'path2vec, batch ' + str(int(batch))
        label = 'path2vec'
    plt.plot(x, y, linestyle='dashed', marker='o', label=label)
plt.xlabel('Vector size')
plt.ylabel('Spearman rank correlation on SimLex999')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance in semantic similarity, static synsets')
# plt.show()
plt.savefig(corpus + '_static_synsets.png', dpi=300)
plt.close()

plt.figure()
plt.plot((50, np.max(vectorsizes)), (graph_score, graph_score), 'red', label='Pure WordNet')
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = dhuman_scores[batchsizes == batch]
    marker = 'o'
    if int(batch) == 166:
        label = 'Deepwalk'
        marker = 'X'
    elif int(batch) == 366:
        label = 'node2vec'
        marker = '*'
    elif int(batch) == 266:
        label = 'TransR'
        marker = 'D'
    elif int(batch) == 265:
        label = 'TransD'
    else:
        #label = 'path2vec, batch ' + str(int(batch))
        label = 'path2vec'
    plt.plot(x, y, linestyle='dashed', marker=marker, label=label)
plt.xlabel('Vector size')
plt.ylabel('Spearman rank correlation on SimLex999')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance in semantic similarity, dynamic synsets')
# plt.show()
plt.savefig(corpus + '_dynamic_synsets.png', dpi=300)
plt.close()
