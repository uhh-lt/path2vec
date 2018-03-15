# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Node Embedding Based on Matrix Factorization Algorithm
"""

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import networkx as nx
import matplotlib as plt
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
"""
Convert Edge List to Adjacency Matrix
"""
filename = 'graph'
embedding_size = 5 # Dimension of the embedding vector.
num_sampled = 4
num_steps = 600
valid_size = 3
num_points = 20
batch_size = 120
valid_examples = np.array(random.sample(range(10), valid_size))
def read_edges(filename):    
    g = nx.read_edgelist(filename, nodetype=str,create_using=nx.DiGraph())
    return g
def build_dataset(graph):
    index = 0
    number_of_edges = graph.number_of_edges()
    dataset = np.ndarray(shape=(number_of_edges), dtype=np.int32)
    labels = np.ndarray(shape=(number_of_edges, 1), dtype=np.int32)
    dictionary = {k: v for v, k in enumerate(graph.nodes)}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    for e in graph.edges:
        dataset[index] = dictionary[e[0]]
        labels[index] = dictionary[e[1]]
        index = index+1
    return dataset, labels, dictionary, reverse_dictionary
G = read_edges(filename)
node_size = G.number_of_nodes()
edges_size = G.number_of_edges()
print("Number of nodes: %d " %node_size)
data,labels,dictionary,rdictionary=build_dataset(G)
graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
    train_dataset = tf.placeholder(tf.int32, shape=[edges_size])
    train_labels = tf.placeholder(tf.int32, shape=[edges_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    embeddings = tf.Variable(
        tf.random_uniform([node_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([node_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([node_size]))
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(\
                weights=softmax_weights,\
                biases=softmax_biases,\
                inputs=embed,\
                labels=train_labels,\
                num_sampled=num_sampled,\
                num_classes=node_size))
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, 
        keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings,tf.transpose(normalized_embeddings))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Variables Initialized......')
    average_loss = 0
    for step in range(num_steps):
        feed_dict = {train_dataset : data, train_labels : labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        print('Average loss at step %d: %f' % (step, l))
        if step % 100== 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = rdictionary[valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = rdictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    pylab.show()

words = [rdictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
