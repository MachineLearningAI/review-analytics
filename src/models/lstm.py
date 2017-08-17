import numpy
import random
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
import nn_utils
import re, string
# fix random seed for reproducibility
numpy.random.seed(7)

# split data into training data and test data
all_data = nn_utils.get_all_labeled_data()
data = all_data["data"]

reviews = []
labels = []
for review in data:
    word_list = review["text"].split()
    for item in word_list:
        exclude = set(string.punctuation)
        item = ''.join(ch for ch in item if ch not in exclude)
    reviews.append(word_list)

    labels.append(review["labels"])


flat_list = [item for sublist in reviews for item in sublist]
num_to_word = dict((k, v) for (k, v) in enumerate(set(flat_list)))
word_to_num = dict((v, k) for (k, v) in enumerate(set(flat_list)))


X_train = []
y_train = []
X_test = []
y_test = []

for review, label in zip(reviews, labels):
    if random.randint(0, 5) > 1:
        X_train.append([word_to_num[item] for item in review])
        # y_train.append(label.index(1))
        y_train.append(label)
    else:
        X_test.append([word_to_num[item] for item in review])
        # y_test.append(label.index(1))
        y_test.append(label)

max_review_length = max([len(X) for X in X_train])
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# y_train = to_categorical(y_train, nb_classes=10)
# y_test = to_categorical(y_train, nb_classes=10)
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(len(flat_list), embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
