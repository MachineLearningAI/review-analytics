
depth = 10

text = open('all_reviews.json', 'r').read()
reviews = eval(text)

print(type(reviews))

for i in range(3):
	year_reviews = reviews[i]

	if i == 0:
		print("\n\nTY14 ----------------------- \n\n")
	elif i == 1:
		print("\n\nTY15 ----------------------- \n\n")
	elif i == 2:
		print("\n\nTY16 ----------------------- \n\n")

	for i in range(depth):
		review = year_reviews[i]
		transcript = review['Review Title'] + ' '
		transcript += review['Review Text'] +' '
		transcript += review['Cons']
		print(transcript)