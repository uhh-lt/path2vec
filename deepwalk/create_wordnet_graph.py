from nltk.corpus import wordnet as wn
from igraph import *
from itertools import islice
import pickle

#init graph
words_graph = Graph()

vertices = []
edges = []
dictionary_id_to_name = {}

#get all noun synsets
noun_synsets = list(wn.all_synsets('n'))
vertices = [synset.name() for synset in noun_synsets]

#add synset names as vertices
words_graph.add_vertices(vertices)

for vertex in words_graph.vs:
    current_vertice_id = vertex.index
    current_vertice_name = vertex['name']
    #if just a noun name, not a full synset name is needed, then use the following line instead
    #current_vertice_name = vertice['name'].split(".")[0]

    #save pair of "id -> name"
    dictionary_id_to_name[str(current_vertice_id)] = current_vertice_name

    #iterate over hypernyms and save new edges.
    # no need to look for hyponyms, as appropriate hyponym will be later/earlier in a main loop and will create connection
    # with current synset as with his hypernym
    current_synset = wn.synset(vertex['name'])
    for hypernym in current_synset.hypernyms():
        hypernym_name = hypernym.name()
        hypernym_id = words_graph.vs.find(hypernym_name).index
        edges.append((current_vertice_id, hypernym_id))

#add edges
words_graph.add_edges(edges)

#save dictionary with "node.index -> name" pairs to use in further computations
# for converting DeepWalk output embedding and saving time on iterating over all WordNet data every time
with open('nodes.pkl', 'wb') as f:
    pickle.dump(dictionary_id_to_name, f, pickle.HIGHEST_PROTOCOL)

#save graph to edge list
words_graph.write_edgelist("wordnet.edgelist")
