#!/usr/bin/python3
# coding: utf-8

from keras.engine import Input
from keras.models import Model
from keras.layers.merge import dot
import gensim
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

m = sys.argv[1]

# Detecting the model format
if m.endswith('.vec.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)
elif m.endswith('.bin.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True, encoding='utf8')
else:
    model = gensim.models.Word2Vec.load(m)

wv = model.wv
embedding_layer = wv.get_keras_embedding()

print(embedding_layer)


input_a = Input(shape=(1,), dtype='int32', name='input_a')
input_b = Input(shape=(1,), dtype='int32', name='input_b')
embedding_a = embedding_layer(input_a)
embedding_b = embedding_layer(input_b)
similarity = dot([embedding_a, embedding_b], axes=2, normalize=True)

keras_model = Model(inputs=[input_a, input_b], outputs=similarity)
keras_model.compile(optimizer='sgd', loss='mse')

print(keras_model.summary())


