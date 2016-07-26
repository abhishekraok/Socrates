from __future__ import print_function
import gensim
from Constants import Constants
import numpy as np
import os


class Word2Vec(object):
    BlankVector = np.zeros((1, Constants.Word2VecConstant))

    def __init__(self, path='../data/GoogleNews-vectors-negative300.bin'):
        if not os.path.isfile(path):
            raise Exception('Word2Vec file ' + path + ' not found')
        print('Initializing Word2Vec from file ', path)
        self.model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        print('Initialization complete')

    def get_vector(self, word):
        """
        Gets vector for particular word. If not found returns the UnknownVector constant

        :type word: str
        :param word: the word for which vector is requested
        :rtype: np.array
        """
        if word in self.model:
            return self.model[word]
        else:
            return Word2Vec.BlankVector

    def get_words(self, vector, n=10):
        return self.model.similar_by_vector(vector, topn=n)[0][0]

    def get_top_word(self, vector):
        """
        if vector is all 0 returns blank  string, if vector is all 1 returns Unknown word
        :param vector: np.array of dimension Word2VecDimension
        :return: word
        """
        if np.all([(i == 0) for i in vector]):
            return ''
        return self.get_words(vector, 1)


if __name__ == "__main__":
    model = Word2Vec()
    word = "congratulations"
    vector = model.get_vector(word)
    print("vector for ", word, ":", vector)
    derived_word = model.get_words(vector, 1)
    print("derived word from the vector:", derived_word)
