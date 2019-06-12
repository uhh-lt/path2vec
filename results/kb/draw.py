#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

vectorsizes = []
batchsizes = []
pearson_scores = []
spearman_scores = []

for line in sys.stdin:
    if line.strip().startswith('#'):
        continue
    res = line.strip().split('\t')
    (corpus, vsize, bsize, pearson, spearman) = res
    if float(bsize) < 100 or float(bsize) > 366 or float(bsize) == 265:
        continue
    vsize = float(vsize)
    if vsize < 50:
        continue
    vectorsizes.append(vsize)
    bsize = float(bsize)
    batchsizes.append(bsize)
    pearson = float(pearson)
    pearson_scores.append(pearson)
    spearman = float(spearman)
    spearman_scores.append(spearman)

diff_bsizes = set(batchsizes)

vectorsizes = np.array(vectorsizes)
batchsizes = np.array(batchsizes)
pearson_scores = np.array(pearson_scores)
spearman_scores = np.array(spearman_scores)

plt.figure()
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = pearson_scores[batchsizes == batch]
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
        label = 'path2vec'
    plt.plot(x, y, linestyle='dashed', marker=marker, label=label)
plt.xlabel('Vector size')
plt.ylabel('Correlation')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance on DBPedia, Pearson score')
plt.savefig('dbp_pearson.png', dpi=300)
plt.close()

plt.figure()
for batch in sorted(diff_bsizes):
    x = vectorsizes[batchsizes == batch]
    y = spearman_scores[batchsizes == batch]
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
        label = 'path2vec'
    plt.plot(x, y, linestyle='dashed', marker=marker, label=label)
plt.xlabel('Vector size')
plt.ylabel('Rank correlation')
plt.legend(loc='best')
plt.grid(True)
plt.title('Models performance on DBPedia, Spearman score')
plt.savefig('dbp_spearman.png', dpi=300)
plt.close()

