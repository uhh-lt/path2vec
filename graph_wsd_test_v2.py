# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:13:25 2018

@author: dorgham
"""

import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import codecs
from sklearn.metrics import f1_score, precision_score, recall_score
import gensim

#algorithm parameters
USE_POS_INFO = True
USE_JCN = False  #if False, lch is used
VECTORIZED_SIMILARITY = False
USE_PAGERANK = False
AVG_METHOD = 'micro'
MAX_DEPTH = 3
senseval_fpath = 'WSD_Unified_Evaluation_Datasets/senseval2/senseval2.data.xml'
gold_tags_fpath = 'WSD_Unified_Evaluation_Datasets/senseval2/senseval2.gold.key.txt'
wn_embedding_fpath = 'jcn-brown-thresh01-near50_embeddings_vsize300_bsize50_lr005.vec'

info_content = wordnet_ic.ic('ic-brown.dat')


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
        model = gensim.models.KeyedVectors.load_word2vec_format(wn_embedding_fpath, binary=False)
    counter=0
    output_dict = dict()
    for index, sentence_ids in enumerate(ids_list):
        G=nx.Graph()
        sent_len = len(sentence_ids)
        G_pos = dict()  #used for aligning the nodes when drawing the graph
        pos_idx=1
        token_nodeNames_map = dict()
        pos_list = poses[index]
        sentence = sentences[index]
        
        #construct the nodes of the graph
        for i, _id in enumerate(sentence_ids):
            if USE_POS_INFO:  #restrict the retrieved snysets from wordnet to the target pos
                wn_pos = convert_to_wordnet_pos(pos_list[i])
            else:
                wn_pos = None
                
            synsets_list = list(wn.synsets(sentence[i], pos=wn_pos))
            if len(synsets_list) > 0:
                node_names = []
                for synset in synsets_list:
                    node_name = str(i) + ' ' + synset.name()
                    #adding the index to the node name is important in the case of 
                    #having a word that is repeated in the sentence but with 
                    #different sense each time, so we want unique node for each one.
                    G.add_node(node_name)
                    node_names.append(node_name)
                token_nodeNames_map[_id] = node_names
                G_pos.update( (label, (pos_idx, j)) for j, label in enumerate(node_names) ) 
                pos_idx+=1
        
        #compute word similarity
        sim_dict = dict()
        for idx, key in enumerate(sentence_ids):
            if USE_POS_INFO:
                wn_pos = convert_to_wordnet_pos(pos_list[idx])
            else:
                wn_pos = None
            synsets_list = list(wn.synsets(sentence[idx], pos=wn_pos))
            if len(synsets_list) > 0:
                i = 1
                while i<=MAX_DEPTH and idx+i<sent_len:
                    if USE_POS_INFO:
                        wn_pos = convert_to_wordnet_pos(pos_list[idx+i])
                    else:
                        wn_pos = None
                        
                    next_synsets_list = list(wn.synsets(sentence[idx+i], pos=wn_pos))
                    if len(next_synsets_list) > 0:
                        for current_synset in synsets_list:
                            for neighbor_synset in next_synsets_list:
                                nodes = str(idx) + ' ' + current_synset.name() + ';'
                                nodes += str(idx+i) + ' ' + neighbor_synset.name()
                                if current_synset.pos() == 'n' and neighbor_synset.pos() == 'n':
                                    if VECTORIZED_SIMILARITY:
                                        sim_dict[nodes] = model.wv.similarity(current_synset.name(), neighbor_synset.name())
                                    else:
                                        if USE_JCN:
                                            sim_dict[nodes] = jcn_similarity(current_synset, neighbor_synset)
                                        else:
                                            sim_dict[nodes] = lch_similarity(current_synset, neighbor_synset)
                    i+=1
        
        #build the edges with the weights
        for key in sim_dict:
            nodeIds = key.split(';')
            G.add_edge(nodeIds[0],nodeIds[1], weight=sim_dict[key])
        
        
        #compute graph centrality
        node_scores = dict()
        if USE_PAGERANK:
            node_scores = nx.pagerank(G)
        else:
            node_scores = G.degree(G.nodes(), "weight")
        
        for token_id in sentence_ids:
            nodeNames = token_nodeNames_map.get(token_id)
            scores = []
            max_label = ""
            wordnet_key = ""
            if nodeNames:
                for nodeName in nodeNames:
                    scores.append(node_scores[nodeName])
                if scores:
                    max_index = max(range(len(scores)), key=scores.__getitem__)
                    max_label = nodeNames[max_index]
            if max_label:
                i = max_label.find(' ')
                lemmas = wn.synset(max_label[i+1:]).lemmas()
                for lemma in lemmas:
                    wordnet_key += lemma.key()+';'
                wordnet_key = wordnet_key[0:-1]
            output_dict[token_id] = wordnet_key
        
        #add the weight as attribute to the nodes of the graph
        #for node in node_scores.keys():
         #   G.node[node]['weight']=node_scores[node]
        
        counter += 1
        if counter==1: #draw the graph of the first sentence
            plt.close()
            nx.draw(G, pos=G_pos, with_labels = True)
            plt.show()
        G.clear()
    
    return output_dict


def load_senseval_data(file_path):
    ids = []
    tokens = []
    pos = []
    all_ids = []
    sentences = []
    pos_list = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for text in root:
        for sentence in text:
            for word in sentence:
                if word.tag == 'instance' and word.attrib['id']: #only include words with the <instance> tag
                    ids.append(word.attrib['id'])
                    tokens.append(word.text)
                    pos.append(word.attrib['pos'])
            if tokens:
                all_ids.append(ids)
                sentences.append(tokens)
                pos_list.append(pos)
                ids = []
                tokens = []
                pos = []
            
    return all_ids, sentences, pos_list



if __name__ == "__main__":
    ids, sents, poses = load_senseval_data(senseval_fpath)
    output_dict = sentence_wsd(ids, sents, poses)
    #load the gold results
    with codecs.open(gold_tags_fpath, 'r', 'utf-8') as f:
        lines = f.readlines()
    wsd_output = []
    gold_output = []
    for line in lines:
        id_key_pair = line.split()
        predicted_keys = output_dict[id_key_pair[0]].split(';')
        gold_keys_set = set(id_key_pair[1:])
        predected_keys_set = set(predicted_keys)
        if len(predected_keys_set.intersection(gold_keys_set)) > 0:
            wsd_output.append(predicted_keys[0])
            gold_output.append(predicted_keys[0])
        else:
            wsd_output.append(predicted_keys[0])
            gold_output.append(id_key_pair[1])
            
    assert len(wsd_output) == len(gold_output)

    f1 = f1_score(gold_output, wsd_output, average=AVG_METHOD)
    precision = precision_score(gold_output, wsd_output, average=AVG_METHOD)
    recall = recall_score(gold_output, wsd_output, average=AVG_METHOD)
    
    print ('F-score: %1.4f' % f1, '  Precision: %1.4f' % precision, '  Recall: %1.4f' % recall)
        
        
        

