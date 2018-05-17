# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:13:25 2018

@author: dorgham
"""

import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import OrderedDict
import codecs
import string
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score

#algorithm parameters
USE_POS_INFO = True
USE_LESK = False
USE_PAGERANK = True
AVG_METHOD = 'micro'
MAX_DEPTH = 3
LESK_NORM_FACTOR = 20 #this value is emperical
senseval_fpath = 'WSD_Unified_Evaluation_Datasets/senseval2/senseval2.data.xml'
gold_tags_fpath = 'WSD_Unified_Evaluation_Datasets/senseval2/senseval2.gold.key.txt'

info_content = wordnet_ic.ic('ic-semcor.dat')
wnlemmatizer = WordNetLemmatizer()
pywsd_stopwords = [u"'s", u"``", u"`"]
STOPWORDS = set(stopwords.words('english') + list(string.punctuation) + pywsd_stopwords)


def lch_similarity(synset1, synset2):
    return wn.lch_similarity(synset1, synset2)
    
def jcn_similarity(synset1, synset2):
    return wn.jcn_similarity(synset1, synset2, info_content)
    
def lesk_similarity(synset1, synset2):
    str1 = str(synset1.definition()).translate(str.maketrans('','',string.punctuation))
    for example in synset1.examples():
        str1 += ' ' + str(example).translate(str.maketrans('','',string.punctuation))
    lemmatized_str1=''
    for word in set(str1.split()):
        lemmatized_str1 += wnlemmatizer.lemmatize(word) + ' '
    for lemma in synset1.lemma_names():
        lemmatized_str1 += ' ' + lemma
    hyper_hypo = set(synset1.hyponyms() + synset1.hypernyms() + synset1.instance_hyponyms() + synset1.instance_hypernyms())
    for hh in hyper_hypo:
        for lemma in hh.lemma_names():
            lemmatized_str1 += ' ' + lemma
    current_set = set(lemmatized_str1.split())
    current_set = set(cs.lower() for cs in current_set)
    current_set = current_set.difference(STOPWORDS)
    #print (current_set)
    str2 = str(synset2.definition()).translate(str.maketrans('','',string.punctuation))
    for example in synset2.examples():
        str2 += ' ' + str(example).translate(str.maketrans('','',string.punctuation))
    lemmatized_str2=''
    for word in set(str2.split()):
        lemmatized_str2 += wnlemmatizer.lemmatize(word) + ' '
    for lemma in synset2.lemma_names():
        lemmatized_str2 += ' ' + lemma
    hyper_hypo = set(synset2.hyponyms() + synset2.hypernyms() + synset2.instance_hyponyms() + synset2.instance_hypernyms())
    for hh in hyper_hypo:
        for lemma in hh.lemma_names():
            lemmatized_str2 += ' ' + lemma
    neighbor_set = set(lemmatized_str2.split())
    neighbor_set = set(ns.lower() for ns in neighbor_set)
    neighbor_set = neighbor_set.difference(STOPWORDS)
    #print (neighbor_set)
    return len(current_set.intersection(neighbor_set))

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

def sentence_wsd(sentences, poses):
    counter=0
    output_dict = dict()
    for sentence in sentences:
        G=nx.Graph()
        sent_len = len(sentence.keys())
        G_pos = dict()  #used for aligning the nodes when drawing the graph
        pos_idx=1
        token_nodeNames_map = dict()
        pos_dict = poses[counter]
        
        #construct the nodes of the graph
        for i, _id in enumerate(sentence.keys()):
            if USE_POS_INFO:  #restrict the retrieved snysets from wordnet to the target pos
                wn_pos = convert_to_wordnet_pos(pos_dict[_id])
            else:
                wn_pos = None
                
            synsets_list = list(wn.synsets(sentence[_id], pos=wn_pos))
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
        ids_list = list(sentence.keys())
        lch_sim_dict = dict()
        jcn_sim_dict = dict()
        lesk_sim_dict = dict()
        #print sentence.values()
        for idx, key in enumerate(ids_list):
            if USE_POS_INFO:
                wn_pos = convert_to_wordnet_pos(pos_dict[ids_list[idx]])
            else:
                wn_pos = None
            synsets_list = list(wn.synsets(sentence[ids_list[idx]], pos=wn_pos))
            if len(synsets_list) > 0:
                i = 1
                while i<=MAX_DEPTH and idx+i<sent_len:
                    if USE_POS_INFO:
                        wn_pos = convert_to_wordnet_pos(pos_dict[ids_list[idx+i]])
                    else:
                        wn_pos = None
                        
                    next_synsets_list = list(wn.synsets(sentence[ids_list[idx+i]], pos=wn_pos))
                    if len(next_synsets_list) > 0:
                        for current_synset in synsets_list:
                            for neighbor_synset in next_synsets_list:
                                nodes = str(idx) + ' ' + current_synset.name() + ';'
                                nodes += str(idx+i) + ' ' + neighbor_synset.name()
                                if current_synset.pos() == 'v' and neighbor_synset.pos() == 'v':
                                    sim_weight = lch_similarity(current_synset, neighbor_synset)
                                    lch_sim_dict[nodes] = sim_weight
                                elif current_synset.pos() == 'n' and neighbor_synset.pos() == 'n':
                                    sim_weight = jcn_similarity(current_synset, neighbor_synset)
                                    jcn_sim_dict[nodes] = sim_weight
                                elif USE_LESK:
                                    sim_weight = lesk_similarity(current_synset, neighbor_synset)
                                    lesk_sim_dict[nodes] = sim_weight
                    i+=1
        
        #normalize the similarity weights and build edges
        if lch_sim_dict:
            max_lch_score = max(lch_sim_dict.values())
            for key in lch_sim_dict:
                nodeIds = key.split(';')
                G.add_edge(nodeIds[0],nodeIds[1], weight=(lch_sim_dict[key]/max_lch_score))
        if jcn_sim_dict:
            max_jcn_score = max(jcn_sim_dict.values())
            for key in jcn_sim_dict:
                nodeIds = key.split(';')
                G.add_edge(nodeIds[0],nodeIds[1], weight=(jcn_sim_dict[key]/max_jcn_score))
        if USE_LESK:
            if lesk_sim_dict:
                max_lesk_score = max(lesk_sim_dict.values())
                if max_lesk_score > 0:
                    for key in lesk_sim_dict:
                        nodeIds = key.split(';')
                        G.add_edge(nodeIds[0],nodeIds[1], weight=(lesk_sim_dict[key]/LESK_NORM_FACTOR))
        
        
        #compute graph centrality
        node_scores = dict()
        if USE_PAGERANK:
            node_scores = nx.pagerank(G)
        else:
            node_scores = G.degree(G.nodes(), "weight")
        
        for token_id in ids_list:
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
    tokens_dict = OrderedDict()
    pos_dict = OrderedDict()
    sentences = []
    pos_list = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    for text in root:
        for sentence in text:
            for word in sentence:
                if word.tag == 'instance' and word.attrib['id']: #only include words with the <instance> tag
                    tokens_dict[word.attrib['id']] = word.text
                    pos_dict[word.attrib['id']] = word.attrib['pos']
            if tokens_dict:
                sentences.append(tokens_dict)
                pos_list.append(pos_dict)
                tokens_dict = dict()
                pos_dict = dict()
            
    return sentences, pos_list



if __name__ == "__main__":
    sents, poses = load_senseval_data(senseval_fpath)
    output_dict = sentence_wsd(sents, poses)
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
        
        
        
