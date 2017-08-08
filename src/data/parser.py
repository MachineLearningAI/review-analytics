import csv
import json

reviews = [] # across all years
attributes = [] # includes all attributes
valid_attributes = ['Review ID', 'Submission Date', 'Initial Publish Data', 'User ID', 
					'Product Name', 'Is a Ratings-Only Review', 'Review Title', 'User Nickname', 
					'User Location', 'User Email Address', 'Shared Review with Facebook', 'Overall Rating',
					'Pros', 'Cons', 'Life Changes (Tags)', 'Review Text', '# of Helpful\nVotes', 
					'# of Not Helpful\nVotes', 'Business', 'Home', 'Kids', 'Language', 'Married',
					'Prior Tax Prep Method', 'Student']
years = [14, 15, 16]

for year in years:
	file_name = "TY" + str(year) + ".csv"
	with open(file_name, 'r') as file:
		reader = csv.reader(file, delimiter=',')
		year_reviews = []
		count = 0
		for row in reader:
			if count == 3:
				for word in row:
					attributes.append(word)
			elif count > 3:
				review = {}
				for i in range(len(row)):
					word = row[i]
					attribute = attributes[i]
					if attribute in valid_attributes:
						review[attribute] = word
				try:
					rating = int(review['Overall Rating'])
					if rating <= 3:
						year_reviews.append(review)
				except (ValueError, TypeError) as err:
					pass

			count += 1
	file.close()
	attributes = []

	print(year, len(year_reviews))

	reviews.append(year_reviews)

with open('all_reviews.json', 'w') as reviews_file:
	json.dump(reviews, reviews_file)







