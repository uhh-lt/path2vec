#!/usr/bin/python3
# coding: utf-8

import sys
import numpy as np
import gzip
from itertools import combinations
from keras.callbacks import Callback
from keras.preprocessing.sequence import skipgrams


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


class Wordpairs(object):
    """
    Reads the gzipped pairs file
    Yields line by line
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with gzip.open(self.filepath, "rb") as rf:
            for line in rf:
                if line.strip():
                    yield line.strip().decode('utf-8')


def build_vocabulary(pairs):
    """
    Generates vocabulary from the sentences
    Counts the total number of training words
    Outputs this number, vocabulary and inverted vocabulary
    """

    vocab = {}
    train_words = 0
    vocab.setdefault('UNK', 0)
    for pair in pairs:
        (word0, word1, similarity) = pair.strip().split('\t')
        for word in [word0, word1]:
            vocab.setdefault(word, 0)
            vocab[word] += 1
            train_words += 1
    print('Vocabulary size = %d' % len(vocab), file=sys.stderr)
    print('Total word tokens in the training set = %d' % train_words, file=sys.stderr)
    sorted_words = sorted(vocab, key=lambda w: vocab[w])
    inv_vocab = []
    reverse_vocab = {}
    index_val = 0
    for word in sorted_words:
        inv_vocab.append(word)
        reverse_vocab[word] = index_val
        index_val += 1
    return train_words, reverse_vocab, inv_vocab


def batch_generator(pairs, vocabulary, vocab_size, nsize):
    """
    Generates training batches
    """
    while True:
        # Iterate over all word pairs
        # For each pair generate word index sequence
        # Produce batches for training
        # Each batch will emit the following things
        # 1 - current word index
        # 2 - context word index
        # 3 - target similarity
        # 4 - the same for negative samples

        for pair in pairs:
            # split the line on tabs
            sequence = pair.strip().split('\t')
            words = sequence[:2]
            sim = np.float64(sequence[2])
            if sim == 0 or sim < 0.03 or sim > 1:
                print(pair, file=sys.stderr)
                continue

            # Convert real words to indexes
            sent_seq = [vocabulary[word] for word in words]

            current_word_index = sent_seq[0]
            context_word_index = sent_seq[1]

            # get negative samples
            neg_samples = get_negative_samples(current_word_index, context_word_index, vocab_size, nsize)

            # Producing a batch containing two positive examples and negative samples
            # batch should be a tuple of inputs and targets
            n_examples = len(neg_samples[1])
            batch = (
                [np.zeros((n_examples, 1), dtype=int), np.zeros((n_examples, 1), dtype=int)], np.zeros((n_examples, 1)))
            for i in range(n_examples):
                batch[0][0][i] = neg_samples[0][i][0]
                batch[0][1][i] = neg_samples[0][i][1]
                pred_sim = neg_samples[1][i]
                if pred_sim != 0:  # if this is a positive example, replace 1 with the real similarity from the file
                    pred_sim = sim
                batch[1][i] = pred_sim
            yield batch


def get_negative_samples(current_word_index, context_word_index, vocab_size, nsize):
    # Generate random negative samples, by default the same number as positive samples
    neg_samples = skipgrams([current_word_index, context_word_index], vocab_size, window_size=1, negative_samples=nsize)
    return neg_samples
