import gensim
class Word2Vec(object):
	#initializer
	# path: path for vector binary dataset
	def __init__(self,path='../data/GoogleNews-vectors-negative300.bin'):
		model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

	# gets vector for particular word
	# word: the word for which vector is requested
	def get_vector(self,word):
		return model[word]

	#gets top_n words for particular vector
	# vector: wordvector n: number of similar words required
	def get_words(self,vector,n=10):
		return model.similar_by_vector(vector,topn=n)
if __name__=="__main__":
	model=Word2Vec()
	word="congratulations"
	vector=model.get_vector(word)
	print "vector for ",word,":",vector
	derived_word=model.get_words(vector,1)
	print "derived word from the vector:",derived_word