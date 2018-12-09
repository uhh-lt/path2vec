# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:13:25 2018

@author: dorgham
"""

import argparse
import codecs
import logging
# import matplotlib.pyplot as plt
import xml.etree.ElementTree as ElementTree
from collections import OrderedDict
from random import choice
import gensim
import networkx as nx
from hamming_cython import hamming_sum
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fse(path):
    model = {}
    for f_line in gensim.utils.smart_open(path):
        f_line = gensim.utils.to_unicode(f_line)
        res = f_line.strip().split('\t')
        (synset, vector) = res
        model[synset] = vector
    return model


def hamming_distance(pair, vec_dic):
    s0 = vec_dic[pair[0]]
    s1 = vec_dic[pair[1]]
    if len(s0) != len(s1):
        raise ValueError()
    distance = hamming_sum(s0, s1) + 0.00001
    return 1 / distance


def lch_similarity(synset1, synset2):
    return wn.lch_similarity(synset1, synset2)


def jcn_similarity(synset1, synset2):
    return wn.jcn_similarity(synset1, synset2, info_content)


def convert_to_wordnet_pos(senseval_pos):
    if senseval_pos == 'VERB':
        return wn.VERB
    elif senseval_pos == 'NOUN':
        return wn.NOUN
    elif senseval_pos == 'ADV':
        return wn.ADV
    elif senseval_pos == 'ADJ':
        return wn.ADJ
    else:
        return None


def sentence_wsd(ids_list, sentences, poses):
    if VECTORIZED_SIMILARITY:
        if 'fse' in wn_embedding_fpath:
            model = load_fse(wn_embedding_fpath)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(wn_embedding_fpath,
                                                                    binary=False)
    else:
        model = None
    counter = 0
    meaningful_cases = 0
    output_dict = OrderedDict()
    for index, sentence_ids in enumerate(ids_list):
        graph = nx.Graph()
        sent_len = len(sentence_ids)
        graph_pos = dict()  # used for aligning the nodes when drawing the graph
        pos_idx = 0
        color_map = []
        token_node_names_map = OrderedDict()
        pos_list = poses[index]
        sentence = sentences[index]

        # construct the nodes of the graph
        for i, _id in enumerate(sentence_ids):
            if USE_POS_INFO:  # restrict the retrieved snysets from wordnet to the target pos
                wn_pos = convert_to_wordnet_pos(pos_list[i])
            else:
                wn_pos = None

            synsets_list = list(wn.synsets(sentence[i], pos=wn_pos))
            if len(synsets_list) > 0:
                node_names = []
                for synset in synsets_list:
                    node_name = str(i) + ' ' + synset.name()
                    # adding the index to the node name is important in the case of
                    # having a word that is repeated in the sentence but with
                    # different sense each time, so we want unique node for each one.
                    graph.add_node(node_name)
                    node_names.append(node_name)
                token_node_names_map[_id] = node_names
                graph_pos.update((label, (pos_idx, j + 1)) for j, label in enumerate(node_names))
                pos_idx += 1

        # compute word similarity
        sim_dict = OrderedDict()
        for idx, key in enumerate(sentence_ids):
            if USE_POS_INFO:
                wn_pos = convert_to_wordnet_pos(pos_list[idx])
            else:
                wn_pos = None
            synsets_list = list(wn.synsets(sentence[idx], pos=wn_pos))
            if len(synsets_list) > 0:
                i = 1
                while i <= MAX_DEPTH and idx + i < sent_len:
                    if USE_POS_INFO:
                        wn_pos = convert_to_wordnet_pos(pos_list[idx + i])
                    else:
                        wn_pos = None

                    next_synsets_list = list(wn.synsets(sentence[idx + i], pos=wn_pos))
                    if len(next_synsets_list) > 0:
                        for current_synset in synsets_list:
                            for neighbor_synset in next_synsets_list:
                                nodes = str(idx) + ' ' + current_synset.name() + ';'
                                nodes += str(idx + i) + ' ' + neighbor_synset.name()
                                if current_synset.pos() == 'n' and neighbor_synset.pos() == 'n':
                                    if VECTORIZED_SIMILARITY:
                                        if 'fse' in wn_embedding_fpath:
                                            sim_dict[nodes] = hamming_distance(
                                                (current_synset.name(), neighbor_synset.name()),
                                                model)
                                        else:
                                            sim_dict[nodes] = model.wv.similarity(
                                                current_synset.name(),
                                                neighbor_synset.name())
                                    else:
                                        if USE_JCN:
                                            sim_dict[nodes] = jcn_similarity(current_synset,
                                                                             neighbor_synset)
                                        else:
                                            sim_dict[nodes] = lch_similarity(current_synset,
                                                                             neighbor_synset)
                    i += 1

        # build the edges with the weights
        for key in sim_dict:
            if sim_dict[key] > threshold:
                node_ids = key.split(';')
                graph.add_edge(node_ids[0], node_ids[1], weight=sim_dict[key])

        # compute graph centrality
        if USE_PAGERANK:
            node_scores = nx.pagerank(graph)
        else:
            node_scores = graph.degree(graph.nodes(), "weight")

        selected_nodes = []
        for token_id in sentence_ids:
            node_names = token_node_names_map.get(token_id)
            scores = []
            max_label = ""
            wordnet_key = ""
            if node_names:
                for nodeName in node_names:
                    scores.append(node_scores[nodeName])
                if scores:
                    if len(set(scores)) > 1 and len(scores) > 1:
                        meaningful_cases += 1
                        max_index = max(range(len(scores)), key=scores.__getitem__)
                    else:
                        if USE_RANDOM:
                            max_index = choice(range(len(scores)))
                        else:
                            max_index = 0
                    max_label = node_names[max_index]
                    selected_nodes.append(max_label)
                else:
                    print(token_id, 'No scores at all')
            if max_label:
                i = max_label.find(' ')
                lemmas = wn.synset(max_label[i + 1:]).lemmas()
                for lemma in lemmas:
                    wordnet_key += lemma.key() + ';'
                wordnet_key = wordnet_key[0:-1]
            output_dict[token_id] = wordnet_key

        for node in graph:
            if node in selected_nodes:
                color_map.append('red')
            else:
                color_map.append('lightblue')

            # add the weight as attribute to the nodes of the graph
        # for node in node_scores.keys():
        #   G.node[node]['weight']=node_scores[node]

    #        counter += 1
    #        if counter==1: #draw the graph of the first sentence
    #            plt.close()
    #            nx.draw_networkx_nodes(graph, pos=graph_pos, node_size=800, node_color=color_map)
    #            labels = {}
    #            for node_name in graph.nodes():
    #                labels[str(node_name)] =str(node_name)
    #            nx.draw_networkx_labels(graph, graph_pos,labels,font_size=13)
    #            weights = nx.get_edge_attributes(graph,'weight')
    #            cnt=0
    #            for i, _id in enumerate(sentence_ids):
    #                if _id in token_node_names_map:
    #                    plt.text(cnt,0, s=sentence[i], horizontalalignment='center',
    # fontsize=10, fontweight='bold')
    #                    cnt += 1
    #            unique_weights = list(set(weights.values()))
    #            for weight in unique_weights:
    #                weighted_edges = [(node1,node2) for (node1,node2,edge_attr)
    # in graph.edges(data=True) if edge_attr['weight']==weight]
    #                width = weight*3
    #                nx.draw_networkx_edges(graph, graph_pos,
    # edgelist=weighted_edges,width=width, edge_color='b')
    #            plt.axis('off')
    #            plt.show()
    #        graph.clear()

    print('Meaningful cases:', meaningful_cases)
    return output_dict


def load_senseval_data(file_path):
    identifiers = []
    tokens = []
    pos = []
    all_ids = []
    sentences = []
    pos_list = []
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    for text in root:
        for sentence in text:
            for word in sentence:
                # only include words with the <instance> tag
                if word.tag == 'instance' and word.attrib['id']:
                    identifiers.append(word.attrib['id'])
                    tokens.append(word.text)
                    pos.append(word.attrib['pos'])
            if tokens:
                all_ids.append(identifiers)
                sentences.append(tokens)
                pos_list.append(pos)
                identifiers = []
                tokens = []
                pos = []

    return all_ids, sentences, pos_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word sense disambiguation')
    parser.add_argument('--model', required=True,
                        help='file with embeddings in word2vec format')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Threshold for edge weights (e.g., 0.8)')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--test_set',
                        help='Which WSD test set to use (senseval2, senseval3, semeval2015)')
    parser.add_argument('--averaging', default='micro',
                        help='averaging type for evaluation (micro, macro, weighted)')
    parser.add_argument('--vectorized', action="store_true", default=True,
                        help='Use vectorized similarity')
    parser.add_argument('--pos', action="store_true", default=True,
                        help='Use POS info')
    parser.add_argument('--random', action="store_true", default=False,
                        help='Use random synset in case all centralities are equal')

    args = parser.parse_args()
    wn_embedding_fpath = args.model
    threshold = args.threshold
    dataset = args.test_set
    senseval_fpath = '../data/senseval/' + dataset + '/' + dataset + '.data.xml'
    gold_tags_fpath = '../data/senseval/' + dataset + '/' + dataset + '.gold.key.txt'
    AVG_METHOD = args.averaging
    VECTORIZED_SIMILARITY = args.vectorized
    USE_POS_INFO = args.pos
    MAX_DEPTH = args.depth
    USE_RANDOM = args.random

    USE_JCN = True  # if False, lch is used
    USE_PAGERANK = False
    info_content = wordnet_ic.ic('ic-semcor.dat')

    ids, sents, poslist = load_senseval_data(senseval_fpath)
    disambiguated = sentence_wsd(ids, sents, poslist)
    # load the gold results
    with codecs.open(gold_tags_fpath, 'r', 'utf-8') as f:
        lines = f.readlines()
    wsd_output = []
    gold_output = []
    for line in lines:
        id_key_pair = line.split()
        predicted_keys = disambiguated[id_key_pair[0]].split(';')
        gold_keys_set = set(id_key_pair[1:])
        predicted_keys_set = set(predicted_keys)
        if len(predicted_keys_set.intersection(gold_keys_set)) > 0:
            wsd_output.append(predicted_keys[0])
            gold_output.append(predicted_keys[0])
        else:
            wsd_output.append(predicted_keys[0])
            gold_output.append(id_key_pair[1])

    assert len(wsd_output) == len(gold_output)

    print('Total predictions:', len(wsd_output))

    f1 = f1_score(gold_output, wsd_output, average=AVG_METHOD)
    precision = precision_score(gold_output, wsd_output, average=AVG_METHOD)
    recall = recall_score(gold_output, wsd_output, average=AVG_METHOD)
    accuracy = accuracy_score(gold_output, wsd_output)

    print('F-score: %1.4f' % f1, '  Precision: %1.4f' % precision,
          '  Recall: %1.4f' % recall, '  Accuracy: %1.4f' % accuracy)
