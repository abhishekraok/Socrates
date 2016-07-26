from __future__ import print_function
import gensim
from Constants import Constants
import numpy as np
import os


class Word2Vec(object):
    UnknownVector = np.zeros((1, Constants.Word2VecConstant))

    def __init__(self, path='../data/GoogleNews-vectors-negative300.bin'):
        if not os.path.isfile(path):
            raise Exception('Word2Vec file ' + path + ' not found')
        self.model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

    def get_vector(self, word):
        """
        Gets vector for particular word. If not found returns the UnknownVector constant

        :type word: str
        :param word: the word for which vector is requested
        :rtype: np.array
        """
        return self.model[word]

    # gets top_n words for particular vector
    # vector: wordvector n: number of similar words required
    def get_words(self, vector, n=10):
        return self.model.similar_by_vector(vector, topn=n)[0][0]

    def get_top_word(self, vector):
        return self.get_words(vector, 1)


if __name__ == "__main__":
    model = Word2Vec()
    word = "congratulations"
    vector = model.get_vector(word)
    print("vector for ", word, ":", vector)
    derived_word = model.get_words(vector, 1)
    print("derived word from the vector:", derived_word)
