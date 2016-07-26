import gensim
class Word2Vec(object):
	#initializer
	# path: path for vector binary dataset
	def __init__(self,path='../data/GoogleNews-vectors-negative300.bin'):
		self.model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

	# gets vector for particular word
	# word: the word for which vector is requested
	def get_vector(self,word):
		try:
			return self.model[word]
		except KeyError:
			return None

	#gets top_n words for particular vector
	# vector: wordvector n: number of similar words required
	def get_words(self,vector,n=10):
		return self.model.similar_by_vector(vector,topn=n)[0][0]
	def get_top_word(self,vector):
		return self.get_words(vector,1)
if __name__=="__main__":
	model=Word2Vec()
	word="congratulations"
	vector=model.get_vector(word)
	#print "vector for ",word,":",vector
	derived_word=model.get_top_word(vector)
	print "derived word from the vector:",derived_word
	print "weird word test"
	weird_word="adadadadadooooo"
	print "vector:",model.get_vector(weird_word)