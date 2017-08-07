import csv

if __name__ == "__main__":

    #Declare constants
	OPTIONS = ["advertising", "fees", "return rejected", "poor/no customer service", "free ver. watered down", "poor explanations", "forced to mail", "too long", "complete"]
	OPTION_STR = " ".join([str(j + 1) + " - " + option + ", " for (j, option) in enumerate(OPTIONS)])[:-1]

	with open('labeled_reviews.csv', 'r+') as review_file:	
		reader = csv.reader(review_file, delimiter='|')
		last_word = None
		for row in reader:
			for word in row:
				last_word = word
	
		if last_word == None:
			start_index = 0
		else:
			start_index = int(last_word)

		review_text = open('reviews_full.json', 'r').read()    
		reviews = eval(review_text)

		reviews = [review for review in reviews if int(review['rating']) <= 3]

		for i in range(start_index, len(reviews)):
			review = reviews[i]
			rating = int(review['rating'])
			transcript = "(" + str(rating) + ") "
			transcript += review['title'] + ' '
			transcript += review['text'] + ' '
			transcript += " ".join(review['cons'])

			print("TRANSCRIPT: " + str(i) + "/" + str(len(reviews)) + "\n", transcript)

			labels = [0 for j in range(len(OPTIONS))]
			complete = False
			while not complete:
				print("OPTIONS: " + str(labels) + "\n", OPTION_STR)
				option_num = int(input())
				labels[option_num - 1] = 1
				complete = OPTIONS[option_num - 1] == "complete"

			print("\n")

			labeled_review = "|".join([transcript] + [str(label) for label in labels] + [str(start_index)])

			review_file.write(labeled_review + "\n")

			start_index += 1


		print("No more reviews!!!!!")