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


review_text = open('reviews_full.json', 'r').read()    
reviews = eval(review_text)

print(reviews[0], "\n")

for review in reviews:
    rating = int(review['rating'])
    if rating <= 3:
        review_text = ""
        review_text += review['title'] + ' '
        review_text += review['text'] + ' '
        for con in review['cons']:
            review_text += con + ' '

        print(review_text, '\n')


# review_len_analysis()

# locations, dates, marrieds, homes, kids, businesses, schools, languages = {}, {}, {}, {}, {}, {}, {}, {}
# fields = [locations, dates, marrieds, homes, kids, businesses, schools, languages]
# field_names = ['location', 'date', 'married', 'home', 'kids', 'business', 'school', 'language']
# field_to_plot = 'language'
# field_labels = []
# field_labels_set = []

# for i in range(len(reviews)):
#     review = reviews[i]
#     for j in range(len(fields)):
#         field = fields[j]
#         review_field = review[field_names[j]]
#         field[review_field] = field.get(review_field, 0) + 1

#         if field_names[j] == field_to_plot:
#             if review_field not in field_labels_set:
#                 field_labels_set.append(review_field)
#                 print(review_field)
#                 print(field_labels_set)
#             field_set_index = field_labels_set.index(review_field)
#             field_labels.append(field_set_index)

# for field in fields:
#     print(field)


# # create transcripts and ratings
# transcripts = []
# ratings = []
# for review in reviews:
#     rating = int(review['rating'])
#     if rating <= 4:
#         ratings.append(rating)

#         transcript = review['text']
#         for con in review['cons']:
#             transcript += ' ' + con
#         transcripts.append(transcript)

# vectorizers = [TfidfVectorizer()]
# colors = ['blue', 'green', 'yellow', 'orange']

# for vectorizer in vectorizers:

#     # vectorize
#     X = vectorizer.fit_transform(transcripts)
#     X = X.toarray()

#     # PCA
#     X = PCA(n_components=2).fit_transform(X)

#     # plot
#     for i in range(len(X)):
#         x, y = X[i, 0], X[i, 1]
#         label = field_labels[i]
#         plt.scatter(x, y, c=colors[label], cmap=plt.cm.Paired, s = 4)

#     # patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
#     # plt.legend(handles=patches, loc=locs[j], fontsize=10)

#     plt.show()
