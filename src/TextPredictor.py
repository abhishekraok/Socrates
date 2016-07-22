import cPickle
import os
import TextPreprocessor

from src.WordMap import WordMap


class TextPredictor():
    def __init__(self, model_file_name, word_map_file):
        if os.path.isfile(model_file_name):
            model = cPickle.load(open(model_file_name, 'rb'))
        else:
            raise Exception("model file not found " + model_file_name)
        self.model = model
        self.word_map = WordMap(dictionary_file_name=word_map_file)

    def get_reply(self, user_text):
        user_word_list = TextPreprocessor.clean_text(user_text)
        numbers = self.word_map.words_to_numbers(user_word_list)
        x_in = TextPreprocessor.numbers_to_one_hot(numbers)
        reply_vector = self.model.predict(x_in)
        reply_numbers = TextPreprocessor.one_hot_to_numbers(reply_vector)
        reply_words = self.word_map.numbers_to_words(reply_numbers)
        return ' '.join(reply_words)
