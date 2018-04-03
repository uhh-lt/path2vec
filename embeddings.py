#!/usr/bin/python3
# coding: utf-8

from keras import backend as K
import numpy as np
from keras import Input
from keras.models import Model
from keras.layers.merge import dot
from keras.layers.embeddings import Embedding
from keras.layers import Lambda
import gzip
import random
import sys


class Wordpairs(object):
    """docstring for Wordpairs"""

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with gzip.open(self.filepath, "rb") as rf:
            for line in rf:
                if line.strip():
                    yield line.strip().decode('utf-8')


def build_vocabulary(pairs):
    # Generating Vocabulary from the sentences
    # Count the total number of training words
    vocab = {}
    train_words = 0
    for pair in pairs:
        (word0, word1, similarity) = pair.strip().split('\t')
        for word in [word0, word1]:
            vocab.setdefault(word, 0)
            vocab[word] += 1
            train_words += 1
    print('Vocabulary size = %d' % len(vocab), file=sys.stderr)
    print('Total words to be trained = %d' % train_words, file=sys.stderr)

    sorted_words = reversed(sorted(vocab, key=lambda w: vocab[w]))
    reverse_vocab = {}
    index_val = 0
    for word in sorted_words:
        reverse_vocab[word] = index_val
        index_val += 1

    return vocab, train_words, reverse_vocab


def batch_generator(pairs, reverse_vocab):
    # Restart running from the first sentence if the the file reading is done
    while True:
        # Read one sentence from the file into memory
        # For each sentence generate word index sequence
        # Now spit out batches for training
        # Each batch will emit the following things
        # 1 - current word index
        # 2 - context word indexes
        # 3 - negative sampled word indexes
        for pair in pairs:
            # split the sentence on whitespace
            words = pair.strip().split('\t')[:2]
            sim = float(pair.strip().split('\t')[2])

            sent_seq = [reverse_vocab[word] for word in words]

            # Create current batch
            current_word_index = np.array([sent_seq[0]])
            context_word_index = np.array([sent_seq[1]])
            # get negative samples
            neg_samples = get_negative_samples(current_word_index)
            # yield a batch here
            # batch should be a tuple of inputs and targets
            yield [current_word_index, context_word_index, neg_samples], [np.array([sim]), np.zeros((1, negative))]


def get_negative_samples(current_word_index):
    # Generate random negative samples
    neg_samples = random.sample(range(vocab_size), negative)
    while current_word_index in neg_samples:
        neg_samples = random.sample(range(vocab_size), negative)
    return np.array([neg_samples])


embedding_dimension = 100
min_count = 2
window_size = 1
negative = 1

trainfile = sys.argv[1]

wordpairs = Wordpairs(trainfile)
vocabulary, no_train_words, reverse_vocabulary = build_vocabulary(wordpairs)
vocab_size = len(vocabulary)

# generate embedding matrices with all values between -1/2d, 1/2d
w_embedding = np.random.uniform(-0.5 / embedding_dimension, 0.5 / embedding_dimension,
                                (vocab_size, embedding_dimension))
c_embedding = np.random.uniform(-0.5 / embedding_dimension, 0.5 / embedding_dimension,
                                (vocab_size, embedding_dimension))

# Model has 3 inputs
# Current word index, context words indexes and negative sampled word indexes
word_index = Input(shape=(1,), name='Word')
context_index = Input(shape=(1,), name='Context')
negative_samples = Input(shape=(negative,), name='Negative')

# All the inputs are processed through embedding layers
word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[w_embedding],
                                 name='Word_embeddings')
context_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[c_embedding],
                                    name='Context_embeddings')
word_embedding = word_embedding_layer(word_index)
context_embedding = context_embedding_layer(context_index)
negative_words_embedding = context_embedding_layer(negative_samples)

final_context = Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dimension,),
                       name='final_context_embedding')(context_embedding)

final_ns = Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dimension,),
                  name='final_ns_embedding')(negative_words_embedding)

# The context is multiplied (dot product) with current word and negative sampled words
word_context_product = dot([word_embedding, final_context], axes=-1, normalize=True,
                           name='word2context')
negative_context_product = dot([word_embedding, final_ns], axes=-1, normalize=True,
                               name='word2negative')

keras_model = Model(inputs=[word_index, context_index, negative_samples],
                    outputs=[word_context_product, negative_context_product])

# binary crossentropy is applied on the output
keras_model.compile(optimizer='adagrad', loss='binary_crossentropy')
print(keras_model.summary())

keras_model.fit_generator(batch_generator(wordpairs, reverse_vocabulary),
                          steps_per_epoch=no_train_words, epochs=10, workers=2)
