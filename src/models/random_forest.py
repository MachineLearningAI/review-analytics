import random
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from math import sqrt

def rf_model():
    percent_training = .70 # proportion of data to use for training

    reviews = get_reviews()

    # shuffle and split reviews
    random.shuffle(reviews)
    training_set = reviews[:int(percent_training * len(reviews))]
    testing_set = reviews[int(percent_training * len(reviews)):]
    training_labels = [row[0] for row in training_set]
    training_data = [row[1] for row in training_set]
    testing_data = [row[1] for row in testing_set]

    # tf-idf vectorize training set
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_data)
    X = X.toarray()

    # tf-idf vectorize testing set
    vectorized_testing_data = [vectorizer.transform([review]) for review in testing_data]
    total = len(vectorized_testing_data)

    # create random forest
    forest = RandomForestClassifier(n_estimators = int(sqrt(len(X[0])))+1)
    forest.fit(X, training_labels)

    # generate and return predictions
    tagged_reviews = []
    for i in range(total):
        tagged_reviews.append([forest.predict(vectorized_testing_data[i])[0], testing_data[i]])

    return tagged_reviews