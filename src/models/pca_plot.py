from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def review_len_analysis():

    title_sum = pros_sum = cons_sum = text_sum = 0
    count = 0

    for i in range(1, 6):
        valid_reviews = [i]
        for review in reviews:
            rating = int(review['rating'])
            if (rating in valid_reviews):
                title_sum += len(review['title'])
                pros_sum += len(review['pros'])
                cons_sum += len(review['cons'])
                text_sum += len(review['text'])
                count += 1

        avg_title_len = str(title_sum / count)[:4]
        avg_pros_len = str(pros_sum / count)[:4]
        avg_cons_len = str(cons_sum / count)[:4]
        avg_text_len = str(text_sum / count)[:4]

        print("rating:", i, "count:", count, "avg_title_len:", avg_title_len, "avg_pros_len:", avg_pros_len, "avg_cons_len:", avg_cons_len, "avg_text_len:", avg_text_len)
    print()

reviews_array = []
review_text = open('reviews.txt', 'r').read()    
reviews = eval(review_text)

print(reviews[0], "\n")

review_len_analysis()

# create transcripts and ratings
transcripts = []
ratings = []
for review in reviews:
    rating = int(review['rating'])
    if rating <= 5:
        ratings.append(rating)

        transcript = review['text']
        for con in review['cons']:
            transcript += ' ' + con
        transcripts.append(transcript)

vectorizers = [TfidfVectorizer()]
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