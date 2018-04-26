#!/usr/bin/python3
# coding: utf-8

import sys
from scipy import stats

if __name__ == '__main__':
    file0 = sys.argv[1]
    file1 = sys.argv[2]

    print('Comparing files', file0, file1)

    similarities0 = []
    similarities1 = []

    for line in open(file0, 'r').readlines():
        res = line.strip().split('\t')
        (word0, word1, sim) = res
        similarities0.append(float(sim))

    for line in open(file1, 'r').readlines():
        res = line.strip().split('\t')
        (word0, word1, sim) = res
        similarities1.append(float(sim))

    corr = stats.spearmanr(similarities0, similarities1)

    print('Correlation:', round(corr[0], 4))
    print('P value:', corr[1])
