#!/usr/bin/python3
# coding: utf-8

import sys
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    for line in sys.stdin:
        res = line.strip().split('\t')
        (offset, vector) = res
        synset = wn.synset_from_pos_and_offset('n', int(offset))
        print(synset.name()+'\t'+vector.strip())
