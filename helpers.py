#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
from numpy import float32 as real
import gzip
from itertools import combinations
from keras.callbacks import Callback
from keras.preprocessing.sequence import skipgrams
from keras import backend
from gensim import utils
import json
import time
from nltk.corpus import wordnet as wn
import random
from smart_open import smart_open

neighbors_dict = dict()
current_pos_samples = [[], []]


def custom_loss(reg_1_output, reg_2_output, beta=0.01, gamma=0.01):
    def my_loss(y_true, y_pred):
        if len(reg_1_output) > 0 and len(reg_2_output) > 0:
            alpha = 1 - (beta + gamma)
            # Mean squared error (main loss):
            m_loss = alpha * backend.mean(backend.square(y_pred - y_true), axis=-1)

            # Auxiliary losses (we are maximizing similarity to neighbors in the graph)
            m_loss -= beta * (sum(reg_1_output) / len(reg_1_output))
            m_loss -= gamma * (sum(reg_2_output) / len(reg_2_output))
        else:
            m_loss = backend.mean(backend.square(y_pred - y_true), axis=-1)

        return m_loss

    return my_loss


def build_neighbors_map(vocab_dict, full_graph=None):
    global neighbors_dict
    neighbor_nodes = []
    for vocab, index in vocab_dict.items():
        if vocab == 'UNK':
            continue
        if full_graph is None and vocab.count('.') < 2:
            continue
        if full_graph:
            neighbors = full_graph.neighbors(vocab)
            for node in neighbors:
                neighbor_nodes.append(vocab_dict[node])
        else:
            synset = wn.synset(vocab)
            hypernyms = synset.hypernyms()
            hyponyms = synset.hyponyms()
            for hypernym in hypernyms:
                if vocab_dict[hypernym.name()]:
                    neighbor_nodes.append(vocab_dict[hypernym.name()])
            for hyponym in hyponyms:
                if vocab_dict[hyponym.name()]:
                    neighbor_nodes.append(vocab_dict[hyponym.name()])

        neighbors_dict[index] = neighbor_nodes
        neighbor_nodes = []

    return neighbors_dict


class Wordpairs(object):
    """
    Reads the gzipped pairs file
    Yields line by line
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with gzip.open(self.filepath, "rb") as pairfile:
            for line in pairfile:
                if line.strip():
                    yield line.strip().decode('utf-8')


def vocab_from_file(vocabfile):
    """
    Generates vocabulary from a vocabulary file in JSON
    Outputs vocabulary and inverted vocabulary
    """

    with smart_open(vocabfile, 'r') as f:
        inv_vocab = json.loads(f.read())
    vocabulary = {}
    for no, word in enumerate(inv_vocab):
        vocabulary[word] = no
    print('Vocabulary size = %d' % len(vocabulary), file=sys.stderr)
    return vocabulary, inv_vocab


def build_vocabulary(pairs):
    """
    Generates vocabulary from the sentences
    Counts the total number of training pairs
    Outputs this number, vocabulary and inverted vocabulary
    """
    vocabulary = {}
    train_pairs = 0
    for pair in pairs:
        (word0, word1, similarity) = pair.split('\t')
        train_pairs += 1
        for word in [word0, word1]:
            vocabulary[word] = 0
    print('Vocabulary size = %d' % len(vocabulary), file=sys.stderr)
    print('Total word pairs in the training set = %d' % train_pairs, file=sys.stderr)
    inv_vocab = sorted(vocabulary.keys())
    inv_vocab.insert(0, 'UNK')
    for word in inv_vocab:
        vocabulary[word] = inv_vocab.index(word)
    return train_pairs, vocabulary, inv_vocab


def batch_generator(pairs, vocabulary, vocab_size, nsize, batch_size, use_neighbors,
                    neighbors_count):
    """
    Generates training batches
    """
    timing = False  # Whether to print out batch generation time
    while True:
        # Iterate over all word pairs
        # For each pair generate word index sequence
        # Produce batches for training
        # Each batch will emit the following things
        # 1 - current word index
        # 2 - context word index
        # 3 - target similarity
        # 4 - the same for negative samples
        samples_per_pair = 2 + 2 * nsize  # How many training instances we get from each pair
        # How many samples will be there in each batch?
        samples_per_batch = samples_per_pair * batch_size

        inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                       np.zeros((samples_per_batch, 1), dtype=int)]
        if use_neighbors:
            for n in range(neighbors_count * 2):
                inputs_list.append(np.zeros((samples_per_batch, 1), dtype=int))
        # Batch should be a tuple of inputs and targets. First we create it empty:
        batch = (inputs_list, np.zeros((samples_per_batch, 1)))
        inst_counter = 0
        start = time.time()
        for pair in pairs:
            # split the line on tabs
            sequence = pair.split('\t')
            words = sequence[:2]
            if words[0] not in vocabulary or words[1] not in vocabulary:
                continue
            sim = np.float64(sequence[2])

            # Convert real words to indexes
            sent_seq = [vocabulary[word] for word in words]

            current_word_index = sent_seq[0]
            context_word_index = sent_seq[1]
            if use_neighbors and neighbors_count > 0:
                w_neighbors = neighbors_dict[current_word_index]
                c_neighbors = neighbors_dict[context_word_index]
                w_nbrs = []
                c_nbrs = []
                for n in range(neighbors_count):
                    if w_neighbors and len(w_neighbors) > n:
                        w_nbrs.append(random.choice(w_neighbors))
                    else:
                        w_nbrs.append(current_word_index)
                    if c_neighbors and len(c_neighbors) > n:
                        c_nbrs.append(random.choice(c_neighbors))
                    else:
                        c_nbrs.append(context_word_index)

            # get negative samples for the current pair
            neg_samples = get_negative_samples(
                current_word_index, context_word_index, vocab_size, nsize)

            # Adding 2 positive examples and the corresponding negative samples to the current batch
            for i in range(samples_per_pair):
                batch[0][0][inst_counter] = neg_samples[0][i][0]
                batch[0][1][inst_counter] = neg_samples[0][i][1]
                if use_neighbors and neighbors_count > 0:
                    for n in range(neighbors_count):
                        batch[0][n + 2][inst_counter] = w_nbrs[n]
                    for n in range(neighbors_count):
                        batch[0][n + neighbors_count + 2][inst_counter] = c_nbrs[n]
                pred_sim = neg_samples[1][i]

                # if this is a positive example, replace 1 with the real similarity from the file:
                if pred_sim != 0:
                    pred_sim = sim
                batch[1][inst_counter] = pred_sim
                inst_counter += 1
            if inst_counter == samples_per_batch:
                yield batch
                end = time.time()
                if timing:
                    print('Batch generation took', end - start, file=sys.stderr)
                inst_counter = 0
                inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                               np.zeros((samples_per_batch, 1), dtype=int)]
                if use_neighbors:
                    for n in range(neighbors_count * 2):
                        inputs_list.append(np.zeros((samples_per_batch, 1), dtype=int))
                batch = (inputs_list, np.zeros((samples_per_batch, 1)))
                start = time.time()


def batch_generator_2(pairs, vocabulary, vocab_size, nsize, batch_size):
    """
    Generates training batches
    """
    global current_pos_samples
    timing = False  # Whether to print out batch generation time

    samples_per_pair = 2 + 2 * nsize  # How many training instances we get from each pair
    # How many samples will be there in each batch?
    samples_per_batch = samples_per_pair * batch_size

    inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                   np.zeros((samples_per_batch, 1), dtype=int)]

    # Batch should be a tuple of inputs and targets. First we create it empty:
    batch = (inputs_list, np.zeros((samples_per_batch, 1)))
    inst_counter = 0
    start = time.time()
    for pair in pairs:
        # split the line on tabs
        sequence = pair.split('\t')
        words = sequence[:2]
        if words[0] not in vocabulary or words[1] not in vocabulary:
            continue
        sim = np.float64(sequence[2])

        # Convert real words to indexes
        sent_seq = [vocabulary[word] for word in words]

        current_word_index = sent_seq[0]
        context_word_index = sent_seq[1]

        current_pos_samples[0].append(current_word_index)
        current_pos_samples[1].append(context_word_index)

        # get negative samples for the current pair
        neg_samples = get_negative_samples(
            current_word_index, context_word_index, vocab_size, nsize)

        # Adding two positive examples and the corresponding negative samples to the current batch
        for i in range(samples_per_pair):
            batch[0][0][inst_counter] = neg_samples[0][i][0]
            batch[0][1][inst_counter] = neg_samples[0][i][1]

            pred_sim = neg_samples[1][i]
            # if this is a positive example, replace 1 with the real similarity from the file:
            if pred_sim != 0:
                pred_sim = sim
            batch[1][inst_counter] = pred_sim
            inst_counter += 1
        if inst_counter == samples_per_batch:
            yield batch
            end = time.time()
            if timing:
                print('Batch generation took', end - start, file=sys.stderr)
            inst_counter = 0
            inputs_list = [np.zeros((samples_per_batch, 1), dtype=int),
                           np.zeros((samples_per_batch, 1), dtype=int)]

            batch = (inputs_list, np.zeros((samples_per_batch, 1)))
            current_pos_samples = [[], []]
            start = time.time()

    # return the remaining samples
    yield batch


def get_node_neighbors(word_idx):
    return neighbors_dict[word_idx]


def get_current_positive_samples():
    return current_pos_samples


def get_negative_samples(current_word_index, context_word_index, vocab_size, nsize):
    # Generate random negative samples, by default the same number as positive samples
    neg_samples = skipgrams([current_word_index, context_word_index], vocab_size, window_size=1,
                            negative_samples=nsize)
    return neg_samples


def save_word2vec_format(fname, vocab, vectors, binary=False):
    """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        vocab : dict
            The vocabulary of words with their ranks
        vectors : numpy.array
            The vectors to be stored
        binary : bool
            If True, the data wil be saved in binary word2vec format, else in plain text.
        """
    if not (vocab or vectors):
        raise RuntimeError('no input')
    total_vec = len(vocab)
    vector_size = vectors.shape[1]
    print('storing %dx%d projection weights into %s' % (total_vec, vector_size, fname))
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8('%s %s\n' % (total_vec, vector_size)))
        position = 0
        for element in sorted(vocab, key=lambda word: vocab[word]):
            row = vectors[position]
            if binary:
                row = row.astype(real)
                fout.write(utils.to_utf8(element) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8('%s %s\n' % (element, ' '.join(repr(val) for val in row))))
            position += 1


class SimilarityCallback(Callback):
    """
    Used to output word similarities (or neighbors) at the end of each epoch
    """

    def __init__(self, validation_model=None):
        super(SimilarityCallback, self).__init__()
        self.validation_model = validation_model

    def on_epoch_end(self, epoch, logs=None):
        pairs = combinations(self.model.vexamples, 2)
        for pair in pairs:
            valid_word0 = pair[0]
            valid_word1 = pair[1]
            sim = self._get_sim_pair(self.model.ivocab.index(valid_word0),
                                     self.model.ivocab.index(valid_word1),
                                     self.validation_model)
            log_str = 'Similarity between %s and %s: %f' % (valid_word0, valid_word1, sim)
            print(log_str)

        # The following code can be used to produce nearest neigbors for validation set:
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
    def _get_sim_pair(valid_word_idx, valid_word_idx2, validation_model):
        """
        Produces similarity between a pair of words, using validation model
        """
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0, ] = valid_word_idx
        in_arr2[0, ] = valid_word_idx2
        out = validation_model.predict_on_batch([in_arr1, in_arr2])
        return out

    @staticmethod
    def _get_sim(valid_word_idx, validation_model, vocab_size):
        """
        Produces similarities to all other words from the vocabulary
        """
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0, ] = valid_word_idx
            in_arr2[0, ] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
