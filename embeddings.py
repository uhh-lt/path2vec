#!/usr/bin/python3
# coding: utf-8

import time
import multiprocessing
from keras import Input
from keras.models import Model
from keras.layers.merge import dot
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras import optimizers
from keras import backend
from helpers import *
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# This script trains word embeddings on pairs of words and their similarities.
# A possible source of such data is Wordnet and its shortest paths.

# Arguments:

# [obligatory] tab-separated gzipped file with training pairs and their similarities:
# person.n.01     lover.n.03       0.22079177574204348
# etc...

# [obligatory] vector size (20, 40, 100, 300, etc)

# [optional] gzipped JSON file with the vocabulary (list of words):
# ["UNK", "'hood.n.01", "1530s.n.01", "15_may_organization.n.01", "1750s.n.01", "1760s.n.01"...]
# etc
# If the vocabulary file is not provided, it will be inferred from the training set
# (can be painfully slow for large datasets)

trainfile = sys.argv[1]  # Gzipped file with pairs and their similarities
embedding_dimension = int(sys.argv[2])  # vector size
negative = 3  # number of negative samples
batch_size = 10  # number of pairs in a batch
cores = multiprocessing.cpu_count()

wordpairs = Wordpairs(trainfile)

if len(sys.argv) < 4:
    print('Building vocabulary from the training set...', file=sys.stderr)
    no_train_pairs, vocabulary, inverted_vocabulary = build_vocabulary(wordpairs)
    print('Building vocabulary finished', file=sys.stderr)
else:
    vocabulary_file = sys.argv[3]  # JSON file with the ready-made vocabulary
    print('Loading vocabulary from file', vocabulary_file, file=sys.stderr)
    vocabulary, inverted_vocabulary = vocab_from_file(vocabulary_file)
    print('Counting the number of pairs in the training set...')
    no_train_pairs = 0
    for line in wordpairs:
        no_train_pairs += 1
    print('Number of pairs in the training set:', no_train_pairs)

vocab_size = len(vocabulary)
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

# For now, let's use Keras defaults for initialization:
word_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1, name='Word_embeddings')

# Model has 2 inputs: current word index, context word index
word_index = Input(shape=(1,), name='Word')
context_index = Input(shape=(1,), name='Context')

# All the inputs are processed through the embedding layer
word_embedding = word_embedding_layer(word_index)
word_embedding = Flatten(name='word_vector')(word_embedding)  # Some Keras black magic for reshaping
context_embedding = word_embedding_layer(context_index)
context_embedding = Flatten(name='context_vector')(context_embedding)  # Some Keras black magic for reshaping

# The current word embedding is multiplied (dot product) with the context word embedding
# TODO: what about negative dot products? Insert sigmoid...?
word_context_product = dot([word_embedding, context_embedding], axes=1, normalize=True, name='word2context')

# Creating the model itself...
keras_model = Model(inputs=[word_index, context_index], outputs=[word_context_product])

# Assigning attributes:
keras_model.vexamples = valid_examples
keras_model.ivocab = inverted_vocabulary
keras_model.vsize = vocab_size

adam = optimizers.Adam()

keras_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])

print(keras_model.summary())
print('Batch size:', batch_size)

# create a secondary validation model to run our similarity checks during training
similarity = dot([word_embedding, context_embedding], axes=1, normalize=True)
validation_model = Model(inputs=[word_index, context_index], outputs=[similarity])
sim_cb = SimilarityCallback(validation_model=validation_model)

steps = no_train_pairs / batch_size  # How many times per epoch we will ask the batch generator to yield a batch

# Let's start training!
start = time.time()
keras_model.fit_generator(batch_generator(wordpairs, vocabulary, vocab_size, negative, batch_size), callbacks=[sim_cb],
                          steps_per_epoch=steps, epochs=20, workers=cores, verbose=2)

end = time.time()
print('Training took:', int(end - start), 'seconds', file=sys.stderr)

# Saving the resulting vectors:
filename = trainfile.split('.')[0]+'_embeddings_'+str(embedding_dimension)+'_'+str(negative)+'.vec.gz'
save_word2vec_format(filename, vocabulary, word_embedding_layer.get_weights()[0])

backend.clear_session()
