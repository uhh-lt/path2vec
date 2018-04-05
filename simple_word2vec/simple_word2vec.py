#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Word2Vec implementation in Tensorflow

"""

import tensorflow as tf
import nltk
import math
import os
import time

import numpy as np
from collections import defaultdict

from tensorflow.contrib.tensorboard.plugins import projector

# You can use tensorflows FLAGS to generate program options with dfefaults, so that you can change parameters from the commandline

tf.flags.DEFINE_integer("num_neg_samples", 4, "Number of negative samples")
tf.flags.DEFINE_integer("steps", 100000, "Number of training steps")
tf.flags.DEFINE_integer("learning_rate", 1.0, "Number of training steps")
tf.flags.DEFINE_float("embedding_size", 100, "Size of the embedding")
tf.flags.DEFINE_boolean("lower_case", True, "Whether the corpus should be lowercased")
tf.flags.DEFINE_boolean("skip_gram", True, "Whether skip gram should be used or CBOW")
tf.flags.DEFINE_integer("min_frequency" , 3  , "Words that occur lower than this frequency are discarded as OOV")

FLAGS = tf.flags.FLAGS

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def build_graph(vocabulary_size, num_sampled, embedding_size, learning_rate):
    # Placeholders for inputs
    contexts = tf.placeholder(tf.int32, shape=[None])
    targets = tf.placeholder(tf.int32, shape=[None, 1])
    
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
    nce_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    embed = tf.nn.embedding_lookup(embeddings, contexts)
    
    # Compute the NCE loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=targets,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    return embeddings, contexts, targets, optimizer, loss

def generate_batch(corpus, batch_size, skip_gram=True):
    
    contexts = np.ndarray(shape=(batch_size*2), dtype=np.int32)
    targets = np.ndarray(shape=(batch_size*2, 1), dtype=np.int32)
    
    for i in range(batch_size):
        random_token_num = int(math.floor(np.random.random_sample() * (len(corpus) -2))) +1
        
        	# E.g. for "the quick brown fox jumped over the lazy dog"
		# (context, target) pairs: ([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox)
		# We can simplify to: (the, quick), (brown, quick), (quick, brown), (fox, brown), ... CBOW
         # => contexts is ids of [the, brown, quick, fox, ...], labels/targets: [quick, quick, brown, brown, ...]
		# (quick, the), (quick, brown), (brown, quick), (brown, fox), ... Skip-gram
         # => contexts and targets reversed
        
        # left context pair
        left = [corpus[random_token_num-1], corpus[random_token_num]]
        # right context pair
        right = [corpus[random_token_num+1], corpus[random_token_num]]
        
        if skip_gram:
            left.reverse()
            right.reverse()
        
        contexts[i*2] = left[0]
        contexts[i*2+1] = right[0]
        
        targets[i*2] = left[1]
        targets[i*2+1] = right[1]
            
    return contexts, targets
    
# load a text file, tokenize it, count occurences and build a word encoder that translate a word into a unique id (sorted by word frequency)    
def load_corpus(filename='t8.shakespeare.txt', lower_case=True, min_frequency=3):
    corpus = []
    
    i=0
    with open(filename, 'r') as in_file:
        for line in in_file:
            if i % 1000 == 0:
                print('Loading ',filename,', processing line',i)
            
            if line[-1]=='\n':
                line = line[:-1]
            line = line.strip()
            if lower_case:
                line = line.lower()
            
            # You need to run nltk.download('punkt') for this to work:
            corpus += nltk.word_tokenize(line)
            
            i+=1
    
    print('compute word encoder...')
    word_counter = defaultdict(int)
    
    for word in corpus:
        word_counter[word] += 1
    
    word_counter = list(word_counter.items())
    word_counter = [elem for elem in word_counter if elem[1] >= min_frequency]
    word_counter.sort(key=lambda x:x[1], reverse=True)
    
    word_encoder = defaultdict(int)
    
    for i,elem in enumerate(word_counter):
        word_encoder[elem[0]] = i
        
    print('done')
    
    return corpus, word_encoder

def train(corpus, word_encoder, vocabulary_size, num_samples, steps):   
    with tf.device('/cpu'):
        with tf.Session() as sess:
            embeddings, contexts, targets, optimizer, loss = build_graph(vocabulary_size, num_samples,
                                                                                  FLAGS.embedding_size, FLAGS.learning_rate)
            
            ## summary ops  
            timestamp = str(int(time.time()))
            train_summary_dir = os.path.join('./', 'w2v_summaries_' + timestamp) + '/'
            ensure_dir(train_summary_dir)
            print('Writing summaries and checkpoints to logdir:' + train_summary_dir)
            model_ckpt_file = os.path.join('./w2v_summaries_'+ timestamp + '/', 'model.ckpt')    
            vocab_file = os.path.join(train_summary_dir, 'metadata.tsv')  

            vocab_items = list(word_encoder.items())
            vocab_items.sort(key=lambda x:x[1])
            print(vocab_items[:100])
            vocab_list = [elem[0] for elem in vocab_items if elem[1] > 0]
            
            with open(vocab_file, 'w') as vocab_file_out:
                vocab_file_out.write('<UNK>'+'\n')
                for word in vocab_list:
                    vocab_file_out.write(word+'\n')
    
            loss_summary = tf.summary.scalar('loss', loss) 
            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = embeddings.name
            embedding.metadata_path = vocab_file  
            train_summary_op = tf.summary.merge_all()
        
            summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            projector.visualize_embeddings(summary_writer, config)

            saver = tf.train.Saver(tf.global_variables())

            ## initalize parameters
            sess.run(tf.global_variables_initializer())
            
            losses = []
            
            ## now do batched SGD training
            for current_step in range(steps):
                inputs, labels = generate_batch(corpus, batch_size=32, skip_gram=FLAGS.skip_gram)
                feed_dict = {contexts: inputs, targets: labels}
                _, cur_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
                
                losses.append(cur_loss)
                             
                if current_step % 100==0 and current_step != 0:
                    summary_str = sess.run(train_summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, current_step)
                    
                if current_step % 1000 == 0:
                    print('step',current_step,'mean loss:', np.mean(np.asarray(losses)))
                    saver.save(sess, model_ckpt_file, current_step)
                    losses = []
                    
            embeddings_np = sess.run(embeddings)
            print('embedding matrix:', embeddings_np)
            
            # implement your neighboor search here


if __name__ == "__main__":                
    corpus, word_encoder = load_corpus(lower_case=FLAGS.lower_case, min_frequency=FLAGS.min_frequency)
    

    corpus_num = [word_encoder[word] for word in corpus]
    
    print('First few tokens of corpus:', corpus[:100])
    print('First few tokens of corpus_num:', list(corpus_num[:100]))
    
    corpus_num = np.asarray(corpus_num)
    
    train(corpus_num, word_encoder, vocabulary_size=max(corpus_num)+1, num_samples=FLAGS.num_neg_samples , steps=FLAGS.steps)
