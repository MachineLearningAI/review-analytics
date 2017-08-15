import random, re, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt
from cnn_utils import get_all_labeled_data
from stop_words import get_stop_words

def find(list, i):
	try:
		return list.index(i)
	except:
		return -1

# stopwords
words_1 = get_stop_words('en')
words_2 = ["to", "the", "and", "it", "was", "for", "my", "is", "of", "that", "you", "me", "in", "as", "do", "with", "at", "be", "your", "on", "this", "its", "use", "but", "all", "have", "there", "had", "so", "or", "if", "will", "way", "can", "get", "which", "did", "very", "don", "too", "an", "been", "when", "didn", "would", "ve", "they", "year", "out", "from", "by", "how", "am"]
words_3 = ["just", "much", "many", "up", "using", "used", "some", "always", "every", "years", "are", "really", "what", "than", "doing", "going"]
stop_words = words_1 + words_2 + words_3

# grab data
#all_data = get_all_labeled_data()['data']
all_data = get_all_labeled_data()['condensed_data']
random.shuffle(all_data)

# preprocess data
texts = []
labels = []
for data in all_data:
	text = re.sub(r'[^\w\s]','', data['text'].lower())
	texts.append(text)
	labels.append(data['labels'])

# split reviews
percent_training = .9
index = int(len(texts) * percent_training)
vectorizer = TfidfVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(texts[:index]).toarray()
y_train = labels[:index]
X_test = vectorizer.transform(texts	[index:]).toarray()
y_test = labels	[index:]

for i in range(len(X_train)):
	elem = X_train[i]
	


# create random forest
n_estimators = 15
forest = RandomForestClassifier(n_estimators = n_estimators)
forest.fit(X_train, y_train)

# predict and score model
predicts = forest.predict(X_test)
correct = 0
for i in range(len(predicts)):
	predict = predicts[i].tolist()
	actual = y_test[i]
	index = find(predict, 1)
	# print(predict)
	if index != -1:
		if actual[index] == 1:
			correct += 1
	else:
		all_zero = True
		for elem in actual:
			if elem != 0:
				all_zero = False
				break
		if all_zero:
			correct += 1

accuracy = correct / len(X_test)
print(n_estimators, ":", "accuracy:", accuracy)