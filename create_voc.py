#!/usr/bin/python3
# coding: utf-8

import sys
import json
from nltk.corpus import wordnet as wn


if __name__ == '__main__':
    synsets = list(wn.all_synsets('n'))
    synsets = sorted([s.name() for s in synsets])
    synsets.insert(0, 'UNK')
    print('Total synsets:', len(synsets), file=sys.stderr)
    output = json.dumps(synsets, ensure_ascii=False)
    print(output)
