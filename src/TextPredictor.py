from __future__ import print_function
import pickle
import os

import TextPreProcessor
from TPModel import TpModel
from EnumsCollection import ModelType


class TextPredictor:
    def __init__(self, model_file_name, word_map_file, history_length):
        if word_map_file:
            self.text_processor = TextPreProcessor.TextPreProcessor(dictionary_file=word_map_file)
        else:
            raise Exception('Word map file needed')
        if model_file_name and os.path.isfile(model_file_name):
            self.model = TpModel.load(model_file_name, model_type=ModelType.FirstLSTMModel,
                                      text_preprocessor=self.text_processor, history_length=history_length)
        else:
            self.model = TpModel(ModelType.FirstLSTMModel, self.text_processor, history_length=history_length)

    def get_reply(self, user_text):
        x_in, y = self.text_processor.text_to_vector(text=user_text, history_length=20)
        reply_vector = self.model.predict(x_in)
        return self.text_processor.vector_to_text(reply_vector)


class DummyTextPredictor(TextPredictor):
    class DummyModel:
        def predict(self, a):
            return a

    def __init__(self, model_file_name, word_map_file):
        self.model = DummyTextPredictor.DummyModel()
        self.text_processor = None


if __name__ == '__main__':
    tp = TextPredictor(model_file_name=None, word_map_file='../data/MostCommon2266.txt', history_length=20)
    for i in range(20):
        print(tp.model.train_on_text_file('../data/small_pride.txt', epochs=1))
        you = 'how are you'
        print('You:', you)
        print('Bot:', tp.get_reply(you))
        today = 'how is the day today'
        print('You:', today)
        print('Bot:', tp.get_reply(today))
