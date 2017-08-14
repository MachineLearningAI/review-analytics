import os
import time
import datetime
from gensim.models.keyedvectors import KeyedVectors
from cnn import CNN
from cnn_utils import get_all_labeled_data
import tensorflow as tf
import numpy as np

EMBEDDING_SIZE = 300
BATCH_SIZE = 50
LABELS = ["Fees/Ads", "Missing/Rejected/eFile", "Customer Service", "State", "Carryover", "UI/UX/Form Error", "Explanations", "Foreign", "Print/Export", "Other"]
NUM_PASSES_PER_FILTER = 100 # From original paper.
FILTER_SIZES = [3, 4, 5] # From original paper.

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

all_data = get_all_labeled_data()
data = all_data['data']
print("Data Size: " + str(len(data)))
input_x = []
input_y = []
vocab = set()
MAX_REVIEW_LENGTH = 10
REVIEW_LENGTH_THRESHOLD = 2500

old_len = len(data)
data = [review for review in data if len(review['text']) < REVIEW_LENGTH_THRESHOLD]
print("Num of Reviews Removed: " + str(old_len - len(data)))

for review in data:
    words = review['text'].split()
    if len(words) > MAX_REVIEW_LENGTH:
        MAX_REVIEW_LENGTH = len(words)
    for word in words:
        vocab.add(word)

print("Longest Review: " + str(MAX_REVIEW_LENGTH))
vocab = list(vocab)
vocab.append("<PAD>")
NUM_WORDS = len(vocab)
print("Vocab Size: " + str(NUM_WORDS))

init_embedding_matrix = np.random.uniform(-1, 1, (NUM_WORDS, EMBEDDING_SIZE))
model = KeyedVectors.load_word2vec_format('~/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
lookup_table = {}
for i in range(len(vocab)):
    lookup_table[vocab[i]] = i
    if vocab[i] in model.wv:
        init_embedding_matrix[i] = model.wv[vocab[i]]

print("Finished word2vec init")

for review in data:
    words = review['text'].split()
    encoding = []
    for word in words:
        encoding.append(lookup_table[word])
    for _ in range(MAX_REVIEW_LENGTH - len(words)):
        encoding.append(lookup_table["<PAD>"])
    input_x.append(encoding)
    input_y.append(review['labels'])
print("Finished encoding")

x_train = input_x[:int(0.9 * len(input_x))]
y_train = input_y[:int(0.9 * len(input_y))]
x_dev = input_x[int(0.9 * len(input_x)):]
y_dev = input_y[int(0.9 * len(input_y)):]

batches = []
for i in range(int(len(x_train) / BATCH_SIZE)):
    batch = [[], []]
    for j in range(i * BATCH_SIZE, ((i + 1) * BATCH_SIZE) - 1):
        batch[0].append(x_train[j])
        batch[1].append(y_train[j])
    batches.append(batch)
print("Finished batching")

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

        # Replacing random init with word2vec embeddings
        sess.run(cnn.embedding_matrix.assign(init_embedding_matrix))

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.x: x_batch,
              cnn.y: y_batch,
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
              cnn.x: x_batch,
              cnn.y: y_batch,
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
            if current_step % 5 == 0:
                dev_step(x_dev, y_dev, writer=dev_summary_writer)

        print("woooo")
