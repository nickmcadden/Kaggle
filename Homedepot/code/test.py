#loop through the ngrams, key_pos will be the number of ngrams or else the one before the first that contains 'with'
from nltk.util import ngrams

title = ['the','cat', 'sat','on', 'the','mat','with','the','dog']

key_pos = 0
for t_ngram in ngrams(title, 2):
	print(t_ngram)
	key_pos += 1
	if t_ngram[-1] == "with":
		key_pos -= 1
		break
	
print key_pos