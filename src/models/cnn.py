import tensorflow as tf
import numpy as np

# CNN-static model from https://arxiv.org/pdf/1408.5882.pdf
# No L2 Regularization because of results from https://arxiv.org/pdf/1510.03820.pdf
# Code adapted from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

class CNN(object):
    def __init__(self, max_review_length, num_labels, num_words, embedding_length, filter_sizes, num_passes_per_filter):
        self.x = tf.placeholder(tf.int32, [None, max_review_length], name="x")
        self.y = tf.placeholder(tf.float32, [None, num_labels], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout") # Disable during evaluation.

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([num_words, embedding_length], -1.0, 1.0), name="embedding_matrix") # Replaced with w2v
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.x) # num reviews x max sentence length x embedding length
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # embedded_chars x 1

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-" + str(i)):
                # Conv layer.
                filter_shape = [filter_size, embedding_length, 1, num_passes_per_filter]
                filter_matrix = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter_matrix")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[num_passes_per_filter]), name="filter_bias")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    filter_matrix,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Nonlinearity.
                h = tf.nn.relu(tf.nn.bias_add(conv, filter_bias), name="h")
                # Max-pooling.
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_review_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pooled")
                pooled_outputs.append(pooled)

        # Flattening result of pooling.
        num_filters_total = num_passes_per_filter * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) # num reviews x (num filters * num passes per filter)

        # Dropout for regularization.
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Getting output.
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_labels]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.nn.sigmoid(self.scores)

        # Calculating average cross-entropy loss.
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses)

        # Calculating average accuracy.
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1)) # Boolean array.
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # correct_predictions = tf.equal(tf.round(self.predictions), self.y)
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
            correct_predictions = tf.reduce_max(tf.multiply(tf.round(self.predictions), self.y), 1)
            self.accuracy = tf.reduce_mean(correct_predictions, name="accuracy")
