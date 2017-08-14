import csv
import re

def get_data_from_labeled(tax_year):
    data = []
    condensed_data = []
    file_name = "labeled_reviews_" + tax_year + ".csv"
    with open(file_name) as f:
        content = f.read()

        tokens = content.split("|")
        asdfjkl = re.findall(r"[\w|]+", content)
        counts = [0 for i in range(20)]
        freq = [0 for i in range(14)]
        for i in range(len(tokens)):
            print(i, tokens[i])
            if i % 17 == 0:
                vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                rating = int(tokens[i][-2])
            elif i % 17 == 1:
                text = tokens[i].strip()
            elif i % 17 == 2:
                ID = tokens[i].strip()
            elif i % 17 >= 3 and i % 17 < 16:
                vector[i % 17 - 3] = int(tokens[i])
            elif i % 17 == 16:
                counts[sum(vector)] += 1
                for j in range(len(vector)):
                    if vector[j] == 1:
                        v = [0 for k in range(len(vector))]
                        v[j] = 1
                        data.append([rating, text, ID, v])
                        print([rating, text, ID, v])
                print("---------------------")
                freq = [i + j for (i, j) in zip(freq, vector)]
                condensed_data.append([rating, text, ID, vector])
            else:
                print("err")
    print("Frequencies of reviews with X # of complaints:", counts)
    print("Frequencies of reviews with X label complaint:", freq)
    print("Vocab size (estimate):", len(set(asdfjkl)))
    return ([e for e in condensed_data if e[3][-1] == 0], [e for e in data if e[3][-1] == 0])
