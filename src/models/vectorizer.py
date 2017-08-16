# input unsplit, lowercased, punctuation-stripped string
def keywords_vec(text):
	keywords = ['free', 'fees', 'expensive', 'efile', 'state', 'rejected', 'charged', 'price', 'charge', 'help', 'phone', 'cost', 'support', 'pay', 'call', 'print', 'filed', 'upgrade', 'return', 'returns', 'customer', 'refund', 'service', 'turbotax', 'information', 'info', 'form', 'like', 'late', 'forms', 'explanations', 'explanation', 'find', 'software']

	words = text.split()
	v = []
	for keyword in keywords:
		if keyword in words:
			v.append(1)
		else:
			v.append(0)	
	return v

# input list of strings
def keywords_vec_from_list(l):
	vs = []
	for text in l:
		vs.append(keywords_vec(text))
	return vs