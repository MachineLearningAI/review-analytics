import tensorflow as tf
import numpy as np

MAX_REVIEW_LENGTH = 10
NUM_WORDS = 100
EMBEDDING_SIZE = 100
LABELS = ['Fees', 'Ads', 'Rejected/missing returns', 'Hard to navigate',
        'Lacking carryover information', 'State form issue', 'Poor explanations',
        'Other countries support issue', 'Print/export problems', 'eFiling',
        'Other', 'None']
NUM_FILTERS = 3
FILTER_SIZES = [2, 3, 5]

x = tf.placeholder(tf.int32, [None, MAX_REVIEW_LENGTH], name="x")
y = tf.placeholder(tf.float32, [None, len(LABELS)], name="y")
dropout = tf.placeholder(tf.float32, name="dropout")

with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([NUM_WORDS, EMBEDDING_SIZE], -1.0, 1.0),
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(FILTER_SIZES):
    with tf.name_scope("maxpool-" + str(i)):
        filter_shape = [filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name="b")

        conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        pooled = tf.nn.max_pool(
            h,
            ksize=[1, MAX_REVIEW_LENGTH - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout)

with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, len(LABELS)], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[len(LABELS)]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=scores, logits=y)
    loss = tf.reduce_mean(losses)

with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
