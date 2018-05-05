#!/usr/bin/python3
# coding: utf-8

import sys


def check(dictionary):
    ready = True
    for el in dictionary:
        if dictionary[el]['count'] < samples_per_bin:
            ready = False
            break
    return ready


# JCN Brown
# Average similarity: 0.046287447128679715
# Standard deviation: 0.01617182070531124
jcn_brown_max = 1.0
jcn_brown_min = 0.034

# JCN SemCor
# Average similarity: 0.057187525906456725
# Standard deviation: 0.06390240650668096
jcn_semcor_max = 1.0
jcn_semcor_min = 0.04249619288783975

# LCH
# Average similarity: 0.9638545055490526
# Standard deviation: 0.26178589818488085
lch_max = 2.9444389791664407
lch_min = 0.11122563511022437

if sys.argv[1] == 'jcn-semcor':
    upper = jcn_semcor_max
    lower = jcn_semcor_min
elif sys.argv[1] == 'jcn-brown':
    upper = jcn_brown_max
    lower = jcn_brown_min
elif sys.argv[1] == 'lch':
    upper = lch_max
    lower = lch_min

samples_per_bin = 1000
no_bins = 10  # We want that many bins

step = (upper - lower) / no_bins

bins = {}
start = lower
for i in range(no_bins):
    bins[i] = {}
    bins[i]['lower'] = start
    bins[i]['upper'] = start + step
    bins[i]['count'] = 0
    start += step

print(bins, file=sys.stderr)

for line in sys.stdin:
    res = line.strip().split('\t')
    (synset0, synset1, sim) = res
    similarity = float(sim)
    for key in bins:
        if bins[key]['lower'] <= similarity < bins[key]['upper']:
            if bins[key]['count'] < samples_per_bin:
                print(line.strip())
                bins[key]['count'] += 1
                break
    state = check(bins)
    if state:
        break

print(bins, file=sys.stderr)
