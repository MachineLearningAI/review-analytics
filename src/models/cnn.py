import tensorflow as tf
import numpy as np
import os
import time
import datetime

MAX_REVIEW_LENGTH = 10
NUM_WORDS = 100
EMBEDDING_SIZE = 100
LABELS = ['Fees', 'Ads', 'Rejected/missing returns', 'Hard to navigate',
        'Lacking carryover information', 'State form issue', 'Poor explanations',
        'Other countries support issue', 'Print/export problems', 'eFiling',
        'Other', 'None']
NUM_PASSES_PER_FILTER = 100
FILTER_SIZES = [3, 4, 5]
x_train = []
y_train = []
x_dev = []
y_dev = []
batches = []

class CNN(object):
    def __init__(self, max_review_length, num_labels, num_words, embedding_length, filter_sizes, num_passes_per_filter):
        self.x = tf.placeholder(tf.int32, [None, max_review_length], name="x")
        self.y = tf.placeholder(tf.float32, [None, num_labels], name="y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout") # Disable during evaluation.

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([num_words, embedding_length], -1.0, 1.0), name="embedding_matrix")
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
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculating average cross-entropy loss.
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses)

        # Calculating average accuracy.
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1)) # Boolean array.
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            max_review_length=MAX_REVIEW_LENGTH,
            num_labels=len(LABELS),
            num_words=NUM_WORDS,
            embedding_length=EMBEDDING_SIZE,
            filter_sizes=FILTER_SIZES,
            num_passes_per_filter=NUM_PASSES_PER_FILTER)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

         # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        for batch in batches:
            x_batch, y_batch = batch[0], batch[1]
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                dev_step(x_dev, y_dev, writer=dev_summary_writer)

        print("woooo")
