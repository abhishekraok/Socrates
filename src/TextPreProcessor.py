from __future__ import print_function
import nltk
import numpy as np
from WordMap import WordMap
from Constants import Constants


class TextPreProcessor:
    Unknown = 'UNK'

    def __init__(self, dictionary_file=None, word_map=None):
        if not dictionary_file and not word_map:
            raise Exception('You must supply either dictionary file or word map')
        if word_map:
            self.word_map = word_map
        else:
            self.word_map = WordMap(dictionary_file_name=dictionary_file)
        self.vocabulary_size = self.word_map.vocabulary_size
        self.word_map.Unknown = TextPreProcessor.Unknown

    @staticmethod
    def create_from_text_file(text_file_name):
        words_list = get_clean_words_from_file(text_file_name, 10 ** 1000)
        word_map = WordMap(words_list=words_list)
        dictionary_txt = 'created_dictionary.txt'
        word_map.save_dictionary(dictionary_txt)
        return TextPreProcessor(word_map=word_map)

    def text_to_vector(self, text, history_length):
        word_list = nltk.word_tokenize(text)
        return self.word_list_to_tensor(word_list, history_length=history_length)

    def vector_to_words(self, vectors):
        return self.one_hot_to_word_list(vectors)

    def word_list_to_tensor(self, words_list, history_length):
        print('Converting words to tensors')
        print('Input is of length ', len(words_list))
        numbers = self.word_map.words_to_numbers(words_list)
        x, y = self.numbers_to_tensor(numbers, history_length=history_length)
        print('Created tensors  of shape ', x.shape, y.shape)
        return x, y

    def one_hot_to_word_list(self, one_hot_matrix):
        numbers = TextPreProcessor.one_hot_to_numbers(one_hot_matrix)
        return self.word_map.numbers_to_words(numbers)

    def numbers_to_tensor(self, number_list, history_length):
        max_value = max(number_list)
        if max_value > self.vocabulary_size:
            raise Exception("The max value is greater than vocabulary size" + str(max_value))
        print('Vectorization...')
        sentences = []
        next_word = []
        for i in range(0, len(number_list) - history_length, Constants.Step):
            sentences.append(number_list[i: i + history_length])
            next_word.append(number_list[i + history_length])
        print('nb sequences:', len(sentences))

        X = np.zeros((len(sentences), history_length, self.vocabulary_size), dtype=np.bool)
        y = np.zeros((len(sentences), self.vocabulary_size), dtype=np.bool)
        for i, word_list in enumerate(sentences):
            for t, word in enumerate(word_list):
                X[i, t, word] = 1
            y[i, next_word[i]] = 1
        return X, y

    @staticmethod
    def one_hot_to_numbers(y):
        return list(np.argmax(y, axis=2).flatten())

    def vector_to_text(self, vector):
        word_list = self.vector_to_words(vectors=vector)
        return ' '.join(word_list)


def get_clean_words_from_file(file, max_input_length):
    with open(file) as opened_file:
        text = opened_file.read()[:max_input_length]
        s = clean_text(text)
        return nltk.word_tokenize(s)


def clean_text(text, max_input_length=10 ** 10000):
    text = text.replace('\n', ' ')
    max_length = min(max_input_length, len(text))
    rotation = np.random.randint(low=0, high=max_length, size=1)[0]
    text = text[rotation:max_length] + text[:rotation]
    # make sure to remove # for category separation
    text = ''.join(e for e in text if e.isalnum() or e in '.?", ')
    return text
