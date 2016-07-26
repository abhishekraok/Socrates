import nltk
import numpy as np

from Word2Vec import Word2Vec


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
        words_list = clean_text_to_words_list(user_text)
        return np.vstack((self.vectorizer.get_vector(i) for i in words_list))

    def matrix_to_line(self, reply_vector):
        return ' '.join(self.vectorizer.get_top_word(i) for i in reply_vector)

    def file_to_tensor(self, conversation_file):
        with open(conversation_file, 'r') as f:
            lines = f.readlines()
            return np.stack((self.line_to_matrix(lines)), axis=0)
