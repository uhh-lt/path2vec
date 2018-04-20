#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
from numpy import float32 as real
import gzip
from itertools import combinations
from keras.callbacks import Callback
from keras.preprocessing.sequence import skipgrams
from gensim import utils
import json
import time


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

    vfile = gzip.open(vocabfile, 'r').read()
    inv_vocab = json.loads(vfile.decode('utf-8'))
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


def batch_generator(pairs, vocabulary, vocab_size, nsize, batch_size):
    """
    Generates training batches
    """
    timing = True  # Whether to print out batch generation time
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
        samples_per_batch = samples_per_pair * batch_size  # How many samples will be there in each batch

        # Batch should be a tuple of inputs and targets. First we create it empty:
        batch = ([np.zeros((samples_per_batch, 1), dtype=int), np.zeros((samples_per_batch, 1), dtype=int)],
                 np.zeros((samples_per_batch, 1)))
        inst_counter = 0
        start = time.time()
        for pair in pairs:
            # split the line on tabs
            sequence = pair.split('\t')
            words = sequence[:2]
            sim = np.float64(sequence[2])

            # Convert real words to indexes
            sent_seq = [vocabulary[word] for word in words]

            current_word_index = sent_seq[0]
            context_word_index = sent_seq[1]

            # get negative samples for the current pair
            neg_samples = get_negative_samples(current_word_index, context_word_index, vocab_size, nsize)

            # Adding two positive examples and the corresponding negative samples to the current batch
            for i in range(samples_per_pair):
                batch[0][0][inst_counter] = neg_samples[0][i][0]
                batch[0][1][inst_counter] = neg_samples[0][i][1]
                pred_sim = neg_samples[1][i]
                if pred_sim != 0:  # if this is a positive example, replace 1 with the real similarity from the file
                    pred_sim = sim
                batch[1][inst_counter] = pred_sim
                inst_counter += 1
            if inst_counter == samples_per_batch:
                yield batch
                end = time.time()
                if timing:
                    print('Batch generation took', end-start, file=sys.stderr)
                inst_counter = 0
                batch = ([np.zeros((samples_per_batch, 1), dtype=int), np.zeros((samples_per_batch, 1), dtype=int)],
                         np.zeros((samples_per_batch, 1)))
                start = time.time()



def get_negative_samples(current_word_index, context_word_index, vocab_size, nsize):
    # Generate random negative samples, by default the same number as positive samples
    neg_samples = skipgrams([current_word_index, context_word_index], vocab_size, window_size=1, negative_samples=nsize)
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
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
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

    def on_epoch_end(self, epoch, logs={}):
        pairs = combinations(self.model.vexamples, 2)
        for pair in pairs:
            valid_word0 = pair[0]
            valid_word1 = pair[1]
            sim = self._get_sim_pair(self.model.ivocab.index(valid_word0), self.model.ivocab.index(valid_word1),
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
        in_arr1[0,] = valid_word_idx
        in_arr2[0,] = valid_word_idx2
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
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
