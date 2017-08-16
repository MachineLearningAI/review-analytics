# input unsplit, lowercased, punctuation-stripped string
def keywords_vec(text):
	keywords = {'free', 'fees', 'expensive', 'efile', 'state', 'rejected', 'charged', 'price', 'charge', 'help', 'phone', 'cost', 'support', 'pay', 'call', 'print', 'filed', 'upgrade', 'return', 'returns', 'customer', 'refund', 'service', 'turbotax', 'information', 'info', 'form', 'like', 'late', 'forms', 'explanations', 'explanation', 'find', 'software'}
	words = text.split()
	v = []
	for word in words:
		if word in keywords:
			v.append(1)
		else:
			v.append(0)	
	return v