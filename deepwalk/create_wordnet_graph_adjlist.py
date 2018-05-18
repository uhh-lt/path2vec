from nltk.corpus import wordnet as wn
import networkx as nx
from itertools import islice
import pickle

G=nx.Graph()

vertices = []
edges = []
dictionary_name_to_id = {}
dictionary_id_to_name = {}

#get all noun synsets
noun_synsets = list(wn.all_synsets('n'))
vertices = [synset.name() for synset in noun_synsets]

G.add_nodes_from(range(0, len(vertices)))

for node in G.nodes():
    G.node[node]['name'] = vertices[node]
    dictionary_name_to_id[vertices[node]] = node
    dictionary_id_to_name[str(node)] = vertices[node]

for node in G.nodes():
    current_vertice_id = node
    current_vertice_name = G.node[node]['name']
    current_synset = wn.synset(current_vertice_name)
    for hypernym in current_synset.hypernyms():
        hypernym_name = hypernym.name()
        hypernym_id = dictionary_name_to_id[hypernym_name]
        edges.append((current_vertice_id, hypernym_id))

G.add_edges_from(edges)
nx.write_adjlist(G, "wordnet.adjlist")

with open('nodes.pkl', 'wb') as f:
    pickle.dump(dictionary_id_to_name, f, pickle.HIGHEST_PROTOCOL)

