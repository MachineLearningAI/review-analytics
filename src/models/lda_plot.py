from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# the highest rating to consider reviews for
RATING_THRESHOLD = 4

reviews_array = []
review_text = open("reviews.txt", "r").read()
reviews = eval(review_text)

# create transcripts and ratings
transcripts = []
ratings = []
for review in reviews:
    rating = int(review['rating'])
    if rating <= RATING_THRESHOLD:
        ratings.append(rating)

        transcript = review['text']
        for con in review['cons']:
            transcript += ' ' + con
        transcripts.append(transcript)

vectorizers = [CountVectorizer, TfidfVectorizer]
colors = ['red', 'orange', 'yellow', 'green', 'blue']


for vectorizer in vectorizers:

    # vectorize
    X = vectorizer.fit_transform(transcripts)
    X = X.toarray()

    # PCA
    X = PCA(n_components=2).fit_transform(X)

    # plot
    for i in range(len(X)):
        x, y = X[i, 0], X[i, 1]
        rating = ratings[i]
        plt.scatter(x, y, c=colors[rating - 1], cmap=plt.cm.Paired, s = 4)

    # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    # plt.legend(handles=patches, loc=locs[j], fontsize=10)

    plt.show()
