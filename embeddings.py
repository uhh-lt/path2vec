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
from helpers import *

# This script trains word embeddings on pairs of words and their similarities.
# A possible source of such data is Wordnet and its shortest paths.

embedding_dimension = 10  # vector size
negative = 1  # number of negative samples
cores = multiprocessing.cpu_count()

trainfile = sys.argv[1]  # Gzipped file with pairs and their similarities

wordpairs = Wordpairs(trainfile)

print('Building vocabulary...', file=sys.stderr)
no_train_words, vocabulary, inverted_vocabulary = build_vocabulary(wordpairs)
vocab_size = len(vocabulary)
print('Building vocabulary finished', file=sys.stderr)

valid_size = 4  # Number of random words to log their nearest neighbours after each epoch
# valid_examples = np.random.choice(vocab_size, valid_size, replace=False)

# But for now we will just use a couple of known pairs to log their similarities:
# Gold similarities:
# measure.n.02    fundamental_quantity.n.01        0.930846519882644
# person.n.01     lover.n.03       0.22079177574204348
valid_examples = ['measure.n.02', 'fundamental_quantity.n.01', 'person.n.01', 'lover.n.03']

# TODO: how to properly initialize weights?
# generate embedding matrix with all values between -0.5d, 0.5d
# w_embedding = np.random.uniform(-0.5 / embedding_dimension, 0.5 / embedding_dimension, (vocab_size, embedding_dimension))
# word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[w_embedding], name='Word_embeddings')

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

# Creatung the model itself...
keras_model = Model(inputs=[word_index, context_index], outputs=[word_context_product])

# Assigning attributes:
keras_model.vexamples = valid_examples
keras_model.ivocab = inverted_vocabulary
keras_model.vsize = vocab_size

adagrad = optimizers.adagrad(lr=0.1)  # Choosing and tuning the optimizer
adam = optimizers.Adam()


keras_model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])

print(keras_model.summary())

# create a secondary validation model to run our similarity checks during training
similarity = dot([word_embedding, context_embedding], axes=1, normalize=True)
validation_model = Model(inputs=[word_index, context_index], outputs=[similarity])
validation_model2 = Model(inputs=[word_index], outputs=[word_embedding])
sim_cb = SimilarityCallback(validation_model=validation_model)

steps = no_train_words/2

# Let's start training!
start = time.time()
keras_model.fit_generator(batch_generator(wordpairs, vocabulary, vocab_size, negative), callbacks=[sim_cb],
                          steps_per_epoch=steps, epochs=10, workers=cores)

end = time.time()
print('Training took:', int(end - start), 'seconds', file=sys.stderr)

# Saving the resulting vectors:
save_word2vec_format('embeddings.vec.gz', vocabulary, word_embedding_layer.get_weights()[0])
