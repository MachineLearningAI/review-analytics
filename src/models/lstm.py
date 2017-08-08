from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import collections
import json

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


with open("reviews.json", "r") as reviews_file:
    reviews = json.loads(reviews_file.read())

transcripts = [(review["text"] + " " + review["title"]).strip() for review in reviews]
# transcripts = [transcripts[i].split() for i in range(len(transcripts))]
transcripts = np.array(transcripts)
transcripts = np.reshape(transcripts, [-1, ])
# print(transcripts[0])
dictionary, reverse = build_dataset(transcripts)

vocab_size = len(dictionary)

# Parameters
n_input = 3
n_hidden = 512
learning_rate = 0.001
training_iters = 50000
display_step = 1000

weights = {
    "out": tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}

biases = {
    "out": tf.Variable(tf.random_normal([vocab_size]))
}

x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
