import gensim
class Word2Vec(Object):
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
	def get_words(self,vector,n):
		return model.similar_by_vector(vector,topn=n)