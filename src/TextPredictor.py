import cPickle
import os

import TextPreProcessor
from src.TPModel import TpModel


class TextPredictor:

    def __init__(self, model_file_name, word_map_file):
        if os.path.isfile(model_file_name):
            self.model = cPickle.load(open(model_file_name, 'rb'))
        else:
            # raise Exception("model file not found " + model_file_name)
            self.model = TpModel()
        if word_map_file is not None:
            self.text_processor = TextPreProcessor.TextPreProcessor(word_map_file)

    def get_reply(self, user_text):
        x_in = self.text_processor.text_to_vector(text=user_text)
        reply_vector = self.model.predict(x_in)
        return self.text_processor.vector_to_text(reply_vector)


class DummyTextPredictor(TextPredictor):
    class DummyModel:
        def predict(self, a):
            return a

    def __init__(self, model_file_name, word_map_file):
        self.model = DummyTextPredictor.DummyModel()
        self.text_processor = TextPreProcessor.TextPreProcessor(word_map_file)


if __name__ == '__main__':
    tp = TextPredictor(model_file_name='lstm_try.p', word_map_file=None)
    print tp.train_words('../data/pride.txt')
