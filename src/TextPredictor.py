import cPickle
import os

from src.WordMap import WordMap



class TextPredictor():
    def __init__(self, model_file_name, dictionary):
        if os.path.isfile(model_file_name):
            model = cPickle.load(open(model_file_name, 'rb'))
        else:
            raise Exception("model file not found " + model_file_name)
        self.model = model
        self.word_map = WordMap()

    def get_reply(self, user_text):
        numbers = self.word_map
