import nltk
import numpy as np
from WordMap import WordMap


class TextPreProcessor:
    Unknown = 'UNK'

    def __init__(self, dictionary_file):
        self.word_map = WordMap(dictionary_file_name=dictionary_file)
        self.word_map.Unknown = TextPreProcessor.Unknown
        pass

    def text_to_vector(self, text):
        word_list = nltk.word_tokenize(text)
        return self.word_list_to_one_hot(word_list)

    def vector_to_words(self, vectors):
        return self.one_hot_to_word_list(vectors)

    def word_list_to_one_hot(self, words_list):
        print 'Converting words to one hot representation'
        numbers = self.word_map.words_to_numbers(words_list)
        return TextPreProcessor.numbers_to_one_hot(numbers)

    def one_hot_to_word_list(self, one_hot_matrix):
        numbers = TextPreProcessor.one_hot_to_numbers(one_hot_matrix)
        return self.word_map.numbers_to_words(numbers)

    @staticmethod
    def numbers_to_one_hot(number_list):
        max_value = max(number_list)
        x = np.zeros([len(number_list), max_value + 1])
        for row, number in enumerate(number_list):
            x[row, number] = 1
        return x

    @staticmethod
    def one_hot_to_numbers(X):
        return list(np.argmax(X, axis=1))


def get_clean_words_from_file(file, max_input_length):
    with open(file) as opened_file:
        text = opened_file.read()[:max_input_length]
        return nltk.word_tokenize(clean_text(text))


def clean_text(text, max_input_length=10 ** 10000):
    text = text.replace('\n', ' ')
    max_length = min(max_input_length, len(text))
    rotation = np.random.randint(low=0, high=max_length, size=1)
    text = text[rotation:max_length] + text[:rotation]
    # make sure to remove # for category separation
    text = ''.join(e for e in text if e.isalnum() or e in '.?", ')
    return text

# if __name__ == '__main__':
#     string = get_clean_words_from_file('../data/pride.txt', 500)
#     print string
#     vecs, vocab = word_list_to_one_hot(string)
#     print vecs
