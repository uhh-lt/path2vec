#!/usr/bin/python3
# coding: utf-8

import multiprocessing
from keras import Input
from keras.models import Model
from keras.layers.merge import dot
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras import optimizers
from keras import regularizers
import sys
from keras.callbacks import TensorBoard, EarlyStopping
import helpers
from tensorflow.python.client import device_lib
import numpy as np
import tensorflow as tf
from keras import backend
import random as rn
import argparse
import time

# This script trains word embeddings on pairs of words and their similarities.
# A possible source of such data is Wordnet and its shortest paths.

# Example cmd for running this script:
# python3 embeddings.py --input_file jcn-semcor.tsv.gz --vsize 300 --bsize 100 --lrate 0.001
# --vocab_file synsets_vocab.json.gz --neighbor_count 3 --use_neighbors True --epochs 15


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning graph embeddings with path2vec')
    parser.add_argument('--input_file', required=True,
                        help='tab-separated gzipped file '
                             'with training pairs and their similarities')
    parser.add_argument('--vsize', type=int, default=300, help='vector size')
    parser.add_argument('--bsize', type=int, default=100, help='batch size')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--vocab_file',
                        help='[optional] gzipped JSON file with the vocabulary (list of words)')
    # If the vocabulary file is not provided, it will be inferred from the training set
    # (can be painfully slow for large datasets)
    parser.add_argument('--fix_seeds', default=True, help='fix seeds to ensure repeatability')
    parser.add_argument('--use_neighbors', action="store_true",
                        help='whether or not to use the neighbor nodes-based regularizer')
    parser.add_argument('--neighbor_count', type=int, default=3,
                        help='number of adjacent nodes to consider for regularization')
    parser.add_argument('--negative_count', type=int, default=3, help='number of negative samples')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--regularize', type=bool, default=False,
                        help='L1 regularization of embeddings')
    parser.add_argument('--name', default='graph_emb',
                        help='Run name, to be used in the file name')
    parser.add_argument('--l1factor', type=float, default=1e-10, help='L1 regularizer coefficient')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='neighbors-based regularizer first coefficient')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='neighbors-based regularizer second coefficient')

    args = parser.parse_args()

    print(device_lib.list_local_devices())

    trainfile = args.input_file  # Gzipped file with pairs and their similarities
    embedding_dimension = args.vsize  # vector size (for example, 20)
    batch_size = args.bsize  # number of pairs in a batch (for example, 10)
    learn_rate = args.lrate  # Learning rate
    neighbors_count = args.neighbor_count
    negative = args.negative_count
    run_name = args.name
    l1_factor = args.l1factor
    beta = args.beta
    gamma = args.gamma

    if args.fix_seeds:
        # fix seeds for repeatability of experiments
        np.random.seed(42)
        rn.seed(12345)
        tf.set_random_seed(2)
        session_conf = \
            tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        backend.set_session(sess)

    cores = multiprocessing.cpu_count()

    wordpairs = helpers.Wordpairs(trainfile)

    if not args.vocab_file:
        print('Building vocabulary from the training set...', file=sys.stderr)
        no_train_pairs, vocab_dict, inverted_vocabulary = helpers.build_vocabulary(wordpairs)
        print('Building vocabulary finished', file=sys.stderr)
    else:
        vocabulary_file = args.vocab_file  # JSON file with the ready-made vocabulary
        print('Loading vocabulary from file', vocabulary_file, file=sys.stderr)
        vocab_dict, inverted_vocabulary = helpers.vocab_from_file(vocabulary_file)
        print('Counting the number of pairs in the training set...')
        no_train_pairs = 0
        for line in wordpairs:
            no_train_pairs += 1
        print('Number of pairs in the training set:', no_train_pairs)

    neighbors_dict = helpers.build_connections(vocab_dict)

    vocab_size = len(vocab_dict)
    valid_size = 4  # Number of random words to log their nearest neighbours after each epoch
    # valid_examples = np.random.choice(vocab_size, valid_size, replace=False)

    # But for now we will just use a couple of known pairs to log their similarities:
    # Gold similarities:
    # measure.n.02    fundamental_quantity.n.01        0.930846519882644
    # person.n.01     lover.n.03       0.22079177574204348
    valid_examples = ['measure.n.02', 'fundamental_quantity.n.01', 'person.n.01', 'lover.n.03']

    if args.regularize:
        word_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1,
                                         name='Word_embeddings',
                                         embeddings_regularizer=regularizers.l1(l1_factor))
    else:
        word_embedding_layer = Embedding(vocab_size, embedding_dimension, input_length=1,
                                         name='Word_embeddings')

    # Model has 2 inputs: current word index, context word index
    word_index = Input(shape=(1,), name='Word')
    context_index = Input(shape=(1,), name='Context')

    w_neighbors_indices = []
    c_neighbors_indices = []

    if args.use_neighbors:
        for n in range(neighbors_count):
            w_neighbors_indices.append(Input(shape=(1,), dtype='int32'))
            c_neighbors_indices.append(Input(shape=(1,), dtype='int32'))

    # All the inputs are processed through the embedding layer
    word_embedding = word_embedding_layer(word_index)
    word_embedding = Flatten(name='word_vector')(word_embedding)
    context_embedding = word_embedding_layer(context_index)
    context_embedding = Flatten(name='context_vector')(context_embedding)
    w_neighbor_embeds = []
    c_neighbor_embeds = []
    if args.use_neighbors:
        for n in range(neighbors_count):
            w_neighbor_embeds.append(Flatten()(word_embedding_layer(w_neighbors_indices[n])))
            c_neighbor_embeds.append(Flatten()(word_embedding_layer(c_neighbors_indices[n])))

    # The current word embedding is multiplied (dot product) with the context word embedding
    word_context_product = dot([word_embedding, context_embedding], axes=1, normalize=True,
                               name='word2context')

    reg1_output = []
    reg2_output = []
    if args.use_neighbors:
        for n in range(neighbors_count):
            reg1_output.append(dot([word_embedding, w_neighbor_embeds[n]], axes=1, normalize=True))
            reg2_output.append(dot([context_embedding, c_neighbor_embeds[n]], axes=1,
                                   normalize=True))

    inputs_list = [word_index, context_index]
    if args.use_neighbors:
        for i in range(neighbors_count):
            inputs_list.append(w_neighbors_indices[i])
        for i in range(neighbors_count):
            inputs_list.append(c_neighbors_indices[i])

    # Creating the model itself...
    keras_model = Model(inputs=inputs_list, outputs=[word_context_product])

    # Assigning attributes:
    keras_model.vexamples = valid_examples
    keras_model.ivocab = inverted_vocabulary
    keras_model.vsize = vocab_size

    adam = optimizers.Adam(lr=learn_rate)

    keras_model.compile(optimizer=adam,
                        loss=helpers.custom_loss(reg1_output, reg2_output, beta, gamma),
                        metrics=['mse'])

    print(keras_model.summary())
    print('Batch size:', batch_size)

    train_name = trainfile.split('.')[0] + '_embeddings_vsize' + str(embedding_dimension) + \
                 '_bsize' + str(batch_size) + '_lr' + str(learn_rate).split('.')[-1] + \
                 '_nn-' + str(args.use_neighbors) + str(args.neighbor_count) + '_reg-' \
                 + str(args.regularize)

    # create a secondary validation model to run our similarity checks during training
    similarity = dot([word_embedding, context_embedding], axes=1, normalize=True)
    validation_model = Model(inputs=[word_index, context_index], outputs=[similarity])
    sim_cb = helpers.SimilarityCallback(validation_model=validation_model)

    loss_plot = TensorBoard(log_dir=train_name + '_logs', write_graph=False)
    earlystopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=1, verbose=1,
                                  mode='auto')

    # How many times per epoch we will ask the batch generator to yield a batch?
    steps = no_train_pairs / batch_size

    # Let's start training!
    start = time.time()
    history = keras_model.fit_generator(
        helpers.batch_generator(wordpairs, vocab_dict, vocab_size, negative, batch_size,
                                args.use_neighbors, neighbors_count),
        callbacks=[sim_cb, loss_plot, earlystopping], steps_per_epoch=steps, epochs=args.epochs,
        workers=cores, verbose=2)

    end = time.time()
    print('Training took:', int(end - start), 'seconds', file=sys.stderr)

    # Saving the resulting vectors:
    filename = train_name + '_' + run_name + '.vec.gz'
    helpers.save_word2vec_format(filename, vocab_dict, word_embedding_layer.get_weights()[0])

    backend.clear_session()
