from __future__ import print_function
import gensim
from Constants import Constants
import numpy as np


class Word2Vec(object):
    UnknownVector = np.zeros((1, Constants.Word2VecConstant))

    def __init__(self, path='../data/GoogleNews-vectors-negative300.bin'):
        """ initializer

        :param path: path for vector binary dataset
        """
        self.model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

    def get_vector(self, word):
        """
        Gets vector for particular word. If not found returns the UnknownVector constant

        :type word: str
        :param word: the word for which vector is requested
        :rtype: np.array
        """
        return self.model.get(word, Word2Vec.UnknownVector)

    # gets top_n words for particular vector
    # vector: wordvector n: number of similar words required
    def get_words(self, vector, n=10):
        return self.model.similar_by_vector(vector, topn=n)


if __name__ == "__main__":
    model = Word2Vec()
    word = "congratulations"
    vector = model.get_vector(word)
    print("vector for ", word, ":", vector)
    derived_word = model.get_words(vector, 1)
    print("derived word from the vector:", derived_word)
