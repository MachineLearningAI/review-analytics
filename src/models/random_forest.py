import random, re, sys, string
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from math import sqrt
from cnn_utils import get_all_labeled_data
from stop_words import get_stop_words
from vectorizer import keywords_vec

def max_index(a):
	max_index = -1
	max = -1
	for i in range(len(a)):
		elem = a[i]
		if elem > max:
			max = elem
			max_index = i
	return max_index

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
TABLE = str.maketrans({key: None for key in string.punctuation})
for data in all_data:
	# text = re.sub(r'[^\w\s]','', data['text'].lower())
	text = data['text'].lower().translate(TABLE)
	texts.append(text)
	labels.append(data['labels'])

# split reviews
percent_training = .90
index = int(len(texts) * percent_training)
vectorizer = CountVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(texts[:index]).toarray().tolist()
for i in range(len(X_train)):
	keywords = keywords_vec(texts[:index][i])
	X_train[i] += keywords
	#print(X_train[i])
	#print(keywords)
	#sys.exit()
y_train = labels[:index]

X_test = vectorizer.transform(texts[index:]).toarray().tolist()
for i in range(len(X_test)):
	keywords = keywords_vec(texts[index:][i])
	X_test[i] += keywords
y_test = labels[index:]

print("COMPLETED")

vocab = {}
for word in vectorizer.vocabulary_:
	index = vectorizer.vocabulary_[word]
	vocab[index] = word

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

# create random forest
n_estimators = 12
forest = RandomForestRegressor(n_estimators = n_estimators, criterion='mse')
forest.fit(X_train, y_train)

# key_words = []
# for i in range(len(forest.feature_importances_)):
# 	gini_impurity = forest.feature_importances_[i]
# 	word = vocab[i]
# 	key_words.append([word, float(gini_impurity)])

# key_words = sorted(key_words, key=lambda key_word: -key_word[1])
# print(key_words[:40])
# for word in key_words[:40]:
# 	print(word[0])


# print(forest.feature_importances_.tolist())

# predict and score model
predicts = forest.predict(X_test)
correct = 0
for i in range(len(predicts)):
	predict = predicts[i].tolist()
	actual = y_test[i]
	index = max_index(predict)
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
	#print(predict, actual)

accuracy = correct / len(X_test)
print(n_estimators, ":", "accuracy:", accuracy)