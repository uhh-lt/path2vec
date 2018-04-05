#!/usr/bin/python3
# coding: utf-8

from keras import backend as K
import numpy as np
from keras import Input
from keras.preprocessing.sequence import skipgrams
from keras.utils import Sequence
from keras.models import Model
from keras.layers.merge import Dot, dot
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.layers import merge, Flatten, Reshape
import gzip
from keras import optimizers
import random
from itertools import combinations
import sys

class SimilarityCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        pairs = combinations(valid_examples, 2)
        for pair in pairs:
            valid_word0 = pair[0]
            valid_word1 = pair[1]
            sim = self._get_sim_pair(inverted_vocabulary.index(valid_word0), inverted_vocabulary.index(valid_word1))
            log_str = 'Similarity between %s and %s: %f' % (valid_word0, valid_word1, float(np.dot(sim[1][0], sim[2][0].T)))
            print(log_str)
            #print('Embeddings of these 2 words:')
            #print(sim[1][0])
            #print(sim[2][0])
        # for i in range(valid_size):
        #     # valid_word = inverted_vocabulary[valid_examples[i]]
        #     valid_word = valid_examples[i]
        #     top_k = 3  # number of nearest neighbors
        #     sim = self._get_sim(inverted_vocabulary.index(valid_word))
        #     nearest = (-sim).argsort()[1:top_k + 1]
        #     log_str = 'Nearest to %s:' % valid_word
        #     for k in range(top_k):
        #         close_word = inverted_vocabulary[nearest[k]]
        #         log_str = '%s %s,' % (log_str, close_word)
        #     print(log_str)
        return

    @staticmethod
    def _get_sim_pair(valid_word_idx, valid_word_idx2):
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        in_arr2[0,] = valid_word_idx2
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        return out

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim


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
    inv_vocab = []
    reverse_vocab = {}
    index_val = 0
    for word in sorted_words:
        inv_vocab.append(word)
        reverse_vocab[word] = index_val
        index_val += 1
    return train_words, reverse_vocab, inv_vocab


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
            current_word_index = sent_seq[0]
            context_word_index = sent_seq[1]
            # get negative samples
            neg_samples = get_negative_samples(current_word_index, context_word_index)
            # Producing instances one by one
            for i in range(len(neg_samples[1])):
                pred_sim = neg_samples[1][i]
                if pred_sim != 0:
                    pred_sim = sim
                batch =[np.array([neg_samples[0][i][0]]), np.array([neg_samples[0][i][1]])], np.array([pred_sim])
                yield batch


            #batch = [np.array([x[0] for x in neg_samples[0]]), np.array([x[1] for x in neg_samples[0]])], \
                    #np.array([x if x==0 else sim for x in neg_samples[1]])
            # yield a batch here
            # batch should be a tuple of inputs and targets
            #yield batch
            #yield [current_word_index, context_word_index, neg_samples], [np.array([sim]), np.zeros((1, negative))]


def get_negative_samples(current_word_index, context_word_index):
    # Generate random negative samples
    neg_samples = skipgrams([current_word_index, context_word_index], vocab_size, window_size=1)
    return neg_samples


embedding_dimension = 10  # vector size
negative = 1  # number of negative samples
valid_size = 4

trainfile = sys.argv[1]

wordpairs = Wordpairs(trainfile)
no_train_words, vocabulary, inverted_vocabulary = build_vocabulary(wordpairs)
vocab_size = len(vocabulary)
#valid_examples = np.random.choice(vocab_size, valid_size, replace=False)
valid_examples = ['measure.n.02', 'fundamental_quantity.n.01', 'person.n.01', 'lover.n.03']

# generate embedding matrices with all values between -0.5d, 0.5d
w_embedding = np.random.uniform(-0.5 / embedding_dimension, 0.5 / embedding_dimension,
                                (vocab_size, embedding_dimension))
c_embedding = np.random.uniform(-0.5 / embedding_dimension, 0.5 / embedding_dimension,
                                (vocab_size, embedding_dimension))

# Model has 2 inputs
# Current word index, context word index
word_index = Input(shape=(1,), name='Word')
context_index = Input(shape=(1,), name='Context')

# All the inputs are processed through embedding layers
#word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[w_embedding],
 #                                name='Word_embeddings')
#context_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[c_embedding],
 #                                   name='Context_embeddings')
# TODO: how to properly initialize weights?
word_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1, name='Word_embeddings')
context_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1, name='Context_embeddings')

word_embedding = word_embedding_layer(word_index)
word_embedding = Flatten(name='word_vector')(word_embedding)
context_embedding = context_embedding_layer(context_index)
context_embedding = Flatten(name='context_vector')(context_embedding)


# The current word is multiplied (dot product) with context word and negative sampled words
word_context_product = dot([word_embedding, context_embedding], axes=1, normalize=True,
                          name='word2context')

similarity = dot([word_embedding, context_embedding], axes=1, normalize=True)

keras_model = Model(inputs=[word_index, context_index], outputs=[word_context_product])

adagrad = optimizers.adagrad(lr=0.1)
keras_model.compile(optimizer=adagrad, loss='mean_squared_error', metrics=['mae'])

# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[word_index, context_index], outputs=[similarity, word_embedding, context_embedding])

sim_cb = SimilarityCallback()
print(validation_model.summary())

keras_model.fit_generator(batch_generator(wordpairs, vocabulary), callbacks=[sim_cb],
                          steps_per_epoch=no_train_words, epochs=200, workers=4)
