#! python3
# coding: utf-8

import sys
import time
from multiprocessing import Pool
from functools import partial
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


def calc_similarity(pair, metrics='lch', infcont=None, printing=True, threshold=0.01):
    syns0 = wn.synset(pair[0])
    syns1 = wn.synset(pair[1])
    if metrics == 'jcn':
        similarity = syns0.jcn_similarity(syns1, infcont)  # Jiang-Conrath
    elif metrics == 'lch':
        similarity = syns0.lch_similarity(syns1)  # Leacock-Chodorow
    elif metrics == 'path':
        similarity = syns0.path_similarity(syns1)  # Shortest path
    elif metrics == 'wup':
        similarity = syns0.wup_similarity(syns1)  # Wu-Palmer
    else:
        return None
    if printing:
        if similarity > threshold:
            print(pair[0] + '\t' + pair[1] + '\t' + str(similarity) + '\n')
    return similarity


def wn_neighbors(synset, debug=False):
    hypernyms = synset.hypernyms()
    hyponyms = synset.hyponyms()
    holonyms = synset.member_holonyms()
    if debug:
        print('Hypernyms:', hypernyms)
        print('Hyponyms:', hyponyms)
        print('Holonyms:', holonyms)
    neighbors = set(hypernyms + hyponyms + holonyms)
    return neighbors


def deep_wn_neigbors(synset, rank=1):
    deep_neighbors = {synset}
    for step in range(rank):
        for found_synset in deep_neighbors:
            new_neighbors = wn_neighbors(found_synset)
            deep_neighbors = deep_neighbors | new_neighbors
    deep_neighbors.discard(synset)
    return deep_neighbors


if __name__ == '__main__':
    method = sys.argv[1]
    ic = None
    if len(sys.argv) > 2:
        corpus = sys.argv[2]
        if corpus == 'semcor':
            ic = wordnet_ic.ic('ic-semcor.dat')
        elif corpus == 'brown':
            ic = wordnet_ic.ic('ic-brown.dat')

    cores = 10  # How many threads to use when generating similarities
    walk_rank = 2  # Order of graph neighbors to consider

    thresholds = {'jcn': 0.1, 'lch': 1.5, 'path': 0.1, 'wup': 0.3}
    sim_threshold = thresholds[method]  # Similarity threshold

    synsets = list(wn.all_synsets('n'))
    print('Total synsets:', len(synsets), file=sys.stderr)
    print('Method:', method, file=sys.stderr)
    if ic:
        print('IC corpus:', corpus, file=sys.stderr)
    print('Cores:', cores, file=sys.stderr)
    print('Neighbour rank:', walk_rank, file=sys.stderr)

    num_neighbours = []
    synset_pairs = set()

    for cur_synset in synsets:
        neighbours = deep_wn_neigbors(cur_synset, rank=walk_rank)
        num_neighbours.append(len(neighbours))
        for neighbour in neighbours:
            synset_pairs.add((cur_synset.name(), neighbour.name()))

    print('Average neighbors:', np.mean(num_neighbours), file=sys.stderr)
    print('Std neighbors:', np.std(num_neighbours), file=sys.stderr)
    print('Total pairs:', len(synset_pairs), file=sys.stderr)

    counter = 0
    start = time.time()

    with Pool(cores) as p:
        func = partial(calc_similarity, metrics=method, infcont=ic, threshold=sim_threshold)
        for i in p.imap_unordered(func, synset_pairs, chunksize=100):
            counter += 1
            if counter % 100000 == 0:
                print(counter, 'out of', len(synset_pairs), file=sys.stderr)

    end = time.time()
    print('Total time spent:', end - start, file=sys.stderr)
