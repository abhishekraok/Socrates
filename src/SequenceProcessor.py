import nltk
import numpy as np

from Word2Vec import Word2Vec
from Constants import Constants


def clean_text_to_words_list(text):
    """
    Cleans the text and allows only some special characters. Converts all to lower case

    :type text:str
    """
    approved_text = ''.join(e for e in text if e.isalnum() or e in '.?", ')
    words_list = nltk.word_tokenize(approved_text.lower())
    return words_list


class SequenceProcessor:
    def __init__(self, word2Vec, words_in_sentence):
        """
        Responsible for converting text to vector and back again using word 2 vec

        :type word2Vec: Word2Vec
        """
        self.vectorizer = word2Vec
        self.words_in_sentence = words_in_sentence

    def line_to_matrix(self, user_text):
        """
        Creates a matrix for given line.
        :param user_text:
        :return: np.array of dimension (words_in_sentence,
        """
        words_list = clean_text_to_words_list(user_text)
        if len(words_list) > self.words_in_sentence:
            print('Warning, length of sentence ', user_text, ' is greater than limit. Will chop it off')
        matrix = np.zeros((self.words_in_sentence, Constants.Word2VecConstant))
        vector_list = (self.vectorizer.get_vector(i) for i in words_list[:self.words_in_sentence])
        for i, vector in enumerate(vector_list):
            matrix[i, :] = vector
        return matrix

    def matrix_to_line(self, reply_vector):
        return ' '.join(self.vectorizer.get_top_word(i) for i in reply_vector if i is not '')

    def file_to_tensor(self, conversation_file):
        with open(conversation_file, 'r') as f:
            lines = f.readlines()
            tensor = np.stack((self.line_to_matrix(line) for line in lines), axis=0)
            print('Created a tensor of shape ', tensor.shape, ' from file ', conversation_file)
            return tensor

    def converstaion_text_file_to_tensor_file(self, conversation_file, output_file_name):
        tensor = self.file_to_tensor(conversation_file)
        np.savetxt(fname=output_file_name, X=tensor, delimiter=',')
