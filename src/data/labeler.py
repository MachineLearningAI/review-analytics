import csv

if __name__ == "__main__":

	YEAR = 16 # UPDATE THIS ACCORDINGLY

	OPTIONS = ["Fees", "Ads", "Missing/Rejected", "Customer Service", "State", "Carryover", "eFile", "UI/UX/Form Err", "Explanations", "Foreign", "Print/Export", "Other", "No Complaint", "complete"]
	OPTION_KEYS = ["`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "Enter"]
	OPTION_STR = " ".join([OPTION_KEYS[j] + ": " + option + ", " for (j, option) in enumerate(OPTIONS)])[:-1]

	file_name = 'labeled_reviews_' + str(YEAR) + '.csv'

	print(file_name)

	with open(file_name, 'r+') as review_file:	
		reader = csv.reader(review_file, delimiter='|')
		last_word = None
		for row in reader:
			for word in row:
				last_word = word
	
		if last_word == None:
			start_index = 0
		else:
			start_index = int(last_word) + 1

		reviews = eval(open('unlabeled_reviews.json', 'r').read())
		reviews = reviews[YEAR - 14]

		for i in range(start_index, len(reviews)):
			review = reviews[i]
			rating = int(review['Overall Rating'])
			transcript = str(rating) + ' | '
			transcript += review['Review Title'] + ' '
			transcript += review['Review Text'] + ' '
			transcript += review['Cons']
			transcript += ' | ' + review['Review ID']

			print("TRANSCRIPT: " + str(i + 1) + "/" + str(len(reviews)) + "\n", transcript)

			labels = [0 for j in range(len(OPTIONS))]
			option_nums = []

			while len(option_nums) == 0:
				print("OPTIONS: " + "\n", OPTION_STR)
				option_nums = [option for option in input() if option in OPTION_KEYS] # parses out any invalid chars

				for option_num in option_nums:
					index = OPTION_KEYS.index(option_num)
					labels[index] = 1

				print("\n")

			labeled_review = "|".join([transcript] + [str(label) for label in labels] + [str(start_index)])

			review_file.write(labeled_review + "\n")

			start_index += 1
