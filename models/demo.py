import random, re, sys, string
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from math import sqrt
from cnn_utils import get_all_labeled_data
from cnn_utils import get_text_and_email_from_id
from nltk.corpus import stopwords
from vectorizer import keywords_vec
import numpy as np

words_1 = stopwords.words('english')
words_2 = ["to", "the", "and", "it", "was", "for", "my", "is", "of", "that", "you", "me", "in", "as", "do", "with", "at", "be", "your",
"on", "this", "its", "use", "but", "all", "have", "there", "had", "so", "or", "if", "will", "way", "can", "get", "which", "did", "very",
"don", "too", "an", "been", "when", "didn", "would", "ve", "they", "year", "out", "from", "by", "how", "am"]
words_3 = ["just", "much", "many", "up", "using", "used", "some", "always", "every", "years", "are", "really", "what", "than", "doing", "going"]
stop_words = words_1 + words_2 + words_3
TABLE = str.maketrans({key: None for key in string.punctuation})

def train():
    all_data = get_all_labeled_data(use_chars=False)['condensed_data']
    random.shuffle(all_data)

    # preprocess data
    texts = []
    labels = []
    for data in all_data:
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

    y_train = labels[:index]

    X_test = vectorizer.transform(texts[index:]).toarray().tolist()
    for i in range(len(X_test)):
    	keywords = keywords_vec(texts[index:][i])
    	X_test[i] += keywords
    y_test = labels[index:]

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
    n_estimators = 10
    forest = RandomForestRegressor(n_estimators = n_estimators, criterion='mse')
    forest.fit(X_train, y_train)
    return all_data, vectorizer, forest

def classify_review():
    labels = ["Fees/Ads", "Missing/Rejected/eFile", "Customer Service", "State", "Carryover", "UI/UX/Form Error", "Explanations", "Foreign", "Print/Export", "Other"]
    print("Training...")
    reviews, vectorizer, model = train()
    while True:
        review_text = input("Enter review text: ")
        vectorized = vectorizer.transform([review_text])
        keyworded = np.append(vectorized.toarray(), np.array(keywords_vec(review_text)))
        prediction = model.predict(keyworded.reshape(1, -1)).reshape(-1, 1)
        review_labels = ""
        for i in range(len(prediction)):
            if prediction[i] != 0:
                review_labels += (labels[i] + " ")
        print(review_labels)


def generate_email_lists():
    labels = ["Fees/Ads", "Missing/Rejected/eFile", "Customer Service", "State", "Carryover", "UI/UX/Form Error", "Explanations", "Foreign", "Print/Export", "Other"]
    print("Training...")
    reviews, vectorizer, model = train()
    while True:
        user_input = input("Enter review ID: ")
        review_text, email = get_text_and_email_from_id(user_input)
        if review_text:
            vectorized = vectorizer.transform([review_text])
            keyworded = np.append(vectorized.toarray(), np.array(keywords_vec(review_text)))
            prediction = model.predict(keyworded.reshape(1, -1)).reshape(-1, 1)
            output = email + ': '
            for i in range(len(prediction)):
                if prediction[i] != 0:
                    output += (labels[i] + " ")
            print(output)
        else:
            print("ERROR: ID not found")
