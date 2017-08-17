import csv
import re

def get_data_from_labeled(tax_year, use_chars):
    data = []
    condensed_data = []
    file_name = "../data/labeled_reviews_" + str(tax_year) + ".csv"
    with open(file_name) as f:
        content = f.read()
        tokens = content.split("|")

        for i in range(len(tokens)):
            if i % 17 == 0:
                vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                rating = int(tokens[i][-2])
            elif i % 17 == 1:
                text = tokens[i].strip()
            elif i % 17 == 2:
                ID = tokens[i].strip()
            elif i % 17 >= 3 and i % 17 < 16:
                vector[i % 17 - 3] = int(tokens[i])
            elif i % 17 == 16 and vector[-2] == 0:
                merged_vector = [0] * 10
                merged_vector[0] = min(1, vector[0] + vector[1]) # Fees/Ads
                merged_vector[1] = min(1, vector[2] + vector[6]) # Missing/Rejected/eFile
                merged_vector[2] = vector[3] # Customer Service
                merged_vector[3] = vector[4] # State
                merged_vector[4] = vector[5] # Carryover/Import
                merged_vector[5] = vector[7] # UI/UX/Form Error
                merged_vector[6] = vector[8] # Explanations
                merged_vector[7] = vector[9] # Foreign
                merged_vector[8] = vector[10] # Print/Export
                merged_vector[9] = vector[11] # Other
                condition = None # Filtering out reviews that are length outliers
                if use_chars:
                    condition = len(text) < 2500
                else:
                    condition = len(text.split()) < 620
                if condition:
                    for j in range(len(merged_vector)):
                        if merged_vector[j] == 1:
                            v = [0 for k in range(len(merged_vector))]
                            v[j] = 1
                            datum = {"rating": rating, "text": text, "ID": ID, "labels": v}
                            data.append(datum)
                    condensed_datum = {"rating": rating, "text": text, "ID": ID, "labels": merged_vector}
                    condensed_data.append(condensed_datum)

    return {"data": data, "condensed_data": condensed_data}


def get_all_labeled_data(use_chars):
    ty14 = get_data_from_labeled(14, use_chars)
    ty15 = get_data_from_labeled(15, use_chars)
    ty16 = get_data_from_labeled(16, use_chars)
    data = ty14["data"] + ty15["data"] + ty16["data"]
    condensed_data = ty14["condensed_data"] + ty15["condensed_data"] + ty16["condensed_data"]
    return {"data": data, "condensed_data": condensed_data}
