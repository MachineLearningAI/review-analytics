from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import collections
import json
import random

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    x = tf.split(x, n_input, 1)
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights["out"]) + biases["out"]


logs_path = 'rnn_words'
write = tf.summary.FileWriter(logs_path)

with open("../data/unlabeled_reviews.json", "r") as reviews_file:
    reviews = eval(reviews_file.read())

transcripts = [(review["text"] + " " + review["title"]).strip() for review in reviews]
# transcripts = [transcripts[i].split() for i in range(len(transcripts))]
transcripts = np.array(transcripts)
transcripts = np.reshape(transcripts, [-1, ])
# print(transcripts[0])
dictionary, reverse = build_dataset(transcripts)

vocab_size = len(dictionary)

# # Parameters
# n_input = 3
# n_hidden = 512
# learning_rate = 0.001
# training_iters = 50000
# display_step = 1000

# weights = {
#     "out": tf.Variable(tf.random_normal([n_hidden, vocab_size]))
# }

# biases = {
#     "out": tf.Variable(tf.random_normal([vocab_size]))
# }

# x = tf.placeholder("float", [None, n_input, 1])
# y = tf.placeholder("float", [None, vocab_size])

# pred = RNN(x, weights, biases)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     step = 0
#     offset = random.randint(0, n_input + 1)
#     end_offset = n_input + 1
#     acc_total = 0
#     loss_total = 0

#     writer.add_graph(session.graph)

#     while step < training_iters:
#         if offset > (len(training_data) - end_offset):
#             offset = random.randint(0, n_input + 1)

#         symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
#         symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

#         symbols_out_onehot = np.zeros([vocab_size], dtype=float)
#         symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
#         symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
