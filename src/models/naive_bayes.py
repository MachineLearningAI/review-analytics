import random, re, sys, string
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from math import sqrt
from cnn_utils import get_all_labeled_data
from stop_words import get_stop_words

# stopwords
words_1 = get_stop_words('en')
words_2 = ["to", "the", "and", "it", "was", "for", "my", "is", "of", "that", "you", "me", "in", "as", "do", "with", "at", "be", "your", "on", "this", "its", "use", "but", "all", "have", "there", "had", "so", "or", "if", "will", "way", "can", "get", "which", "did", "very", "don", "too", "an", "been", "when", "didn", "would", "ve", "they", "year", "out", "from", "by", "how", "am"]
words_3 = ["just", "much", "many", "up", "using", "used", "some", "always", "every", "years", "are", "really", "what", "than", "doing", "going"]
stop_words = words_1 + words_2 + words_3

# grab data
all_data = get_all_labeled_data()['data']
all_data_condensed = get_all_labeled_data()['condensed_data']
condensed_data_map = {}
for data in all_data_condensed:
	condensed_data_map[data['ID']] = data['labels']
random.shuffle(all_data)

# preprocess data
texts = []
labels = []
IDs = []
TABLE = str.maketrans({key: None for key in string.punctuation})
for data in all_data:
	text = data['text'].lower().translate(TABLE)
	texts.append(text)
	labels.append(data['labels'].index(1))
	IDs.append(data['ID'])

# split reviews 
percent_training = .90
index = int(len(texts) * percent_training)
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(texts[:index]).toarray().tolist()
y_train = labels[:index]
ID_train = IDs[:index]
X_test = vectorizer.transform(texts	[index:]).toarray().tolist()
y_test = labels[index:]
ID_test = IDs[index:]

try:
	for i in range(len(X_train)):
		x = X_train[i]
		y = y_train[i]
		if sum(y) == 0:
			X_train.remove(x)
			y_train.remove(y)
except:
	pass

try:
	for i in range(len(X_test)):
		x = X_test[i]
		y = y_test[i]
		if sum(y) == 0:
			X_test.remove(x)
			y_test.remove(y)
except:
	pass

# create and fit naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# predict and score model
predicts = nb.predict(X_test)
correct = 0
for i in range(len(predicts)):
	predict = predicts[i]
	ID = ID_test[i]
	actual_vector = condensed_data_map[ID]
	for j in range(len(actual_vector)):
		elem = actual_vector[j]
		if elem == 1:
			if j == predict:
				correct += 1
				break

accuracy = correct / len(X_test)
print("accuracy:", accuracy)
