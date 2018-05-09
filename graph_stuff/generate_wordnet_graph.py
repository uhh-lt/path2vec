#!/usr/bin/python3
# coding: utf-8

from igraph import *
from nltk.corpus import wordnet as wn


synsets = list(wn.all_synsets('n'))

synsets = [s.name() for s in synsets]

g = Graph()

g.add_vertices(synsets)

g.vs["name"] = synsets

print(g.summary())

edges = []
edge_properties = []

for node in g.vs:
    identifier = node.index
    synset = wn.synset(node['name'])
    hypernyms = synset.hypernyms()
    for hyp in hypernyms:
        hyp_name = hyp.name()
        hyp_identifier = g.vs.find(hyp_name).index
        edge = (identifier, hyp_identifier)
        edges.append(edge)
        edge_properties.append('hyponym')

g.add_edges(edges)
g.es['type'] = edge_properties
g.vs["label"] = g.vs["name"]
print(g.summary())


# g.write_graphmlz('wordnet_nltk.graphmlz')
