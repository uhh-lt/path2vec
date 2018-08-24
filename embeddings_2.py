#!/usr/bin/python3
# coding: utf-8

import multiprocessing
from keras import Input, losses
from keras.models import Model
from keras.layers.merge import dot
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras import optimizers
from keras import backend
from keras.callbacks import TensorBoard, EarlyStopping
from helpers import *
from tensorflow.python.client import device_lib
import numpy as np
import tensorflow as tf
import keras.backend as K
import random as rn


np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(2)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


Neighbors_count = 3

#print(device_lib.list_local_devices())

    
def myLoss(reg_1_output, reg_2_output):
    def customLoss(y_true, y_pred):
        beta = 10e-3
        gamma = 10e-3
        alpha = 1 - (beta+gamma)
        m_loss = alpha * K.mean(K.square(y_pred - y_true), axis=-1)
        
        m_loss -= beta * (sum(reg_1_output) / float(len(reg_1_output)))
        m_loss -= gamma * (sum(reg_2_output) / float(len(reg_2_output)))
    
        return m_loss
    return customLoss

# This script trains word embeddings on pairs of words and their similarities.
# A possible source of such data is Wordnet and its shortest paths.

# Arguments:

# [obligatory] tab-separated gzipped file with training pairs and their similarities:
# person.n.01     lover.n.03       0.22079177574204348
# etc...

# [obligatory] vector size (20, 40, 100, 300, etc)

# [obligatory] batch size (10, 20, 60, 100 etc)

# [obligatory] learning rate (default for Adam is 0.001)

# [optional] gzipped JSON file with the vocabulary (list of words):
# ["UNK", "'hood.n.01", "1530s.n.01", "15_may_organization.n.01", "1750s.n.01", "1760s.n.01"...]
# etc
# If the vocabulary file is not provided, it will be inferred from the training set
# (can be painfully slow for large datasets)

trainfile = sys.argv[1]  # Gzipped file with pairs and their similarities
embedding_dimension = int(sys.argv[2])  # vector size (for example, 20)
batch_size = int(sys.argv[3])  # number of pairs in a batch (for example, 10)
learn_rate = float(sys.argv[4])   # Learning rate

negative = 3  # number of negative samples
cores = multiprocessing.cpu_count()

wordpairs = Wordpairs(trainfile)

if len(sys.argv) < 6:
    print('Building vocabulary from the training set...', file=sys.stderr)
    no_train_pairs, vocab_dict, vocab_list = build_vocabulary(wordpairs)
    print('Building vocabulary finished', file=sys.stderr)
else:
    vocabulary_file = sys.argv[5]  # JSON file with the ready-made vocabulary
    print('Loading vocabulary from file', vocabulary_file, file=sys.stderr)
    vocab_dict, vocab_list = vocab_from_file(vocabulary_file)
    print('Counting the number of pairs in the training set...')
    no_train_pairs = 0
    for line in wordpairs:
        no_train_pairs += 1
    print('Number of pairs in the training set:', no_train_pairs)

neighbors_dict = build_connections(vocab_dict)

vocab_size = len(vocab_dict)
valid_size = 4  # Number of random words to log their nearest neighbours after each epoch
# valid_examples = np.random.choice(vocab_size, valid_size, replace=False)

# But for now we will just use a couple of known pairs to log their similarities:
# Gold similarities:
# measure.n.02    fundamental_quantity.n.01        0.930846519882644
# person.n.01     lover.n.03       0.22079177574204348
valid_examples = ['measure.n.02', 'fundamental_quantity.n.01', 'person.n.01', 'lover.n.03']

# TODO: how to properly initialize weights?
# generate embedding matrix with all values between -0.5d, 0.5d
# w_embedding = np.random.uniform(
# -0.5 / embedding_dimension, 0.5 / embedding_dimension, (vocab_size, embedding_dimension))
# word_embedding_layer = Embedding(
# input_dim=vocab_size, output_dim=embedding_dimension, weights=[w_embedding], name='Word_embeddings')

# Model has 2 inputs: current word index, context word index
word_index = Input(shape=(1,), name='Word', dtype='int32')
context_index = Input(shape=(1,), name='Context', dtype='int32')

w_neighbors_indices = []
c_neighbors_indices = []

for n in range(Neighbors_count):
    w_neighbors_indices.append(Input(shape=(1,), dtype='int32'))
    c_neighbors_indices.append(Input(shape=(1,), dtype='int32'))


# For now, let's use Keras defaults for initialization:
word_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1, name='Word_embeddings')

# All the inputs are processed through the embedding layer
word_embedding = word_embedding_layer(word_index)
word_embedding = Flatten(name='word_vector')(word_embedding)  # Some Keras black magic for reshaping
context_embedding = word_embedding_layer(context_index)
context_embedding = Flatten(name='context_vector')(context_embedding)  # Some Keras black magic for reshaping
w_neighbor_embeds = []
c_neighbor_embeds = []
for n in range(Neighbors_count):
    w_neighbor_embeds.append(Flatten()(word_embedding_layer(w_neighbors_indices[n])))
    c_neighbor_embeds.append(Flatten()(word_embedding_layer(c_neighbors_indices[n])))


# The current word embedding is multiplied (dot product) with the context word embedding
# TODO: what about negative dot products? Insert sigmoid...?
word_context_product = dot([word_embedding, context_embedding], axes=1, normalize=True, name='word2context')

# create a secondary validation model to run our similarity checks during training
similarity = dot([word_embedding, context_embedding], axes=1, normalize=True)
validation_model = Model(inputs=[word_index, context_index], outputs=[similarity])

reg1_output = []
reg2_output = []
for n in range(Neighbors_count):
    reg1_output.append(dot([word_embedding, w_neighbor_embeds[n]], axes=1, normalize=True))
    reg2_output.append(dot([context_embedding, c_neighbor_embeds[n]], axes=1, normalize=True))

inputs_list = [word_index, context_index]
for i in range(Neighbors_count):
    inputs_list.append(w_neighbors_indices[i])
for i in range(Neighbors_count):
    inputs_list.append(c_neighbors_indices[i])
# Creating the model itself...
keras_model = Model(inputs= inputs_list, outputs=[word_context_product])

# Assigning attributes:
keras_model.vexamples = valid_examples
keras_model.ivocab = vocab_list
keras_model.vsize = vocab_size

# TODO:  increase the batch size during training, as in https://openreview.net/pdf?id=B1Yy1BxCZ ?

adam = optimizers.Adam(lr=learn_rate)

keras_model.compile(optimizer=adam, loss=myLoss(reg1_output, reg2_output), metrics=['mse'])

print(keras_model.summary())
print('Batch size:', batch_size)

train_name = trainfile.split('.')[0] + '_embeddings_vsize' \
             + str(embedding_dimension)+'_bsize'+str(batch_size)+'_lr'+str(learn_rate).split('.')[-1]


sim_cb = SimilarityCallback(validation_model=validation_model)

loss_plot = TensorBoard(log_dir=train_name + '_logs', write_graph=False, embeddings_freq=10)
earlystopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=1, verbose=1, mode='auto')

steps = no_train_pairs / batch_size  # How many times per epoch we will ask the batch generator to yield a batch

# Let's start training!
start = time.time()
history = keras_model.fit_generator(batch_generator(wordpairs, vocab_dict, vocab_size, negative, batch_size, Neighbors_count),
                                    callbacks=[sim_cb, loss_plot, earlystopping], steps_per_epoch=steps, epochs=10,
                                    workers=cores, verbose=1)

end = time.time()
print('Training took:', int(end - start), 'seconds', file=sys.stderr)

# Saving the resulting vectors:
filename = train_name + '.vec.gz'
save_word2vec_format(filename, vocab_dict, word_embedding_layer.get_weights()[0])

backend.clear_session()
