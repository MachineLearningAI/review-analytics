from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from time import time

# the highest rating to consider reviews for
RATING_THRESHOLD = 5
METHOD_NAME = "lda"
NUM_TOPICS = 8 # LDA
NUM_COMPONENTS = 2 # PCA
LEARNING_DECAY = 0.7 # should be in the interval (0.5, 1.0]
MAX_ITER = 10
NUM_TOP_WORDS = 20


# http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


reviews_array = []
review_text = open('../data/reviews.json', 'r').read()
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

stop_words = ["to", "the", "and", "it", "was", "for", "my", "is", "of", "that", "you", "me", "tax", "turbo", "turbotax", "taxes", "in", "as", "do", "with", "at", "be", "your", "on", "this", "its", "use", "but", "all", "have", "there", "had", "so", "or", "if", "will", "way", "can", "get", "which", "did", "very", "don", "too", "an", "been", "when", "didn", "would", "ve", "they", "file", "filing", "year", "out", "from", "by", "how", "am"]
optional_stop_words = ["easy", "not", "just", "much", "many", "only", "up", "using", "used", "free", "some", "always", "every", "years", "are", "really", "what", "than", "none", "no"]
stop_words.extend(optional_stop_words)

vectorizers = [TfidfVectorizer(stop_words=stop_words)]
colors = ['red', 'orange', 'yellow', 'green', 'blue']

for vectorizer in vectorizers:

    # vectorize
    X = vectorizer.fit_transform(transcripts)
    X = X.toarray()

    if METHOD_NAME == "pca":
        # PCA
        X = PCA(n_components=NUM_COMPONENTS).fit_transform(X)
        # plot
        for i in range(len(X)):
            x, y = X[i, 0], X[i, 1]
            rating = ratings[i]
            plt.scatter(x, y, c=colors[rating - 1], cmap=plt.cm.Paired, s = 4)
            plt.show()
            # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            # plt.legend(handles=patches, loc=locs[j], fontsize=10)
    elif METHOD_NAME == "lda":
        # LDA
        # Griffiths TL, Steyvers M (2004)
        lda = LDA(n_topics=NUM_TOPICS,
                  doc_topic_prior=50 / NUM_TOPICS,
                  topic_word_prior=0.1,
                  learning_decay=LEARNING_DECAY,
        )
        t0 = time()
        lda.fit_transform(X)
        print("lda fit done in %0.3fs" % (time() - t0))
        print_top_words(lda, vectorizer.get_feature_names(), NUM_TOP_WORDS)
