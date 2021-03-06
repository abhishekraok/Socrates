from __future__ import print_function

import os

from keras.models import model_from_json

from socrates.Constants import Constants
from socrates.EnumsCollection import ModelType
from socrates.ModelFactory import ModelFactory
from socrates.unused import TextPreProcessor


class TpModel():
    """
    The offline class to train models, the interface against which the online service calls.
    This is a general text prediction model and specific details are contained in self.model.
    """

    def __init__(self, model_type, text_preprocessor, history_length):
        self.history_length = history_length
        self.text_processor = text_preprocessor
        self.model = ModelFactory.get_model(model_type, input_shape=(history_length, text_preprocessor.vocabulary_size), nb_classes=text_preprocessor.vocabulary_size)

    def predict(self, x_in):
        self.model.predict(x_in)

    def train_on_text_file(self, text_file_name, epochs):
        print('Training on ', text_file_name)
        text = TextPreProcessor.get_clean_words_from_file(text_file_name, 10 ** 7)
        x, y = self.text_processor.word_list_to_tensor(text, history_length=self.history_length)
        print('Shape of X ', x.shape, ' shape of y ', y.shape)
        self.model.fit(x, y, nb_epoch=epochs)
        print('Training done')
        return self.model.evaluate(x, y)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    @staticmethod
    def get_full_file_names(file_name):
        return (file_name + '.json', file_name + '.h5')

    def save(self, file_name):
        json_string = self.model.to_json()
        json_file_name, h5_file_name = TpModel.get_full_file_names(file_name)
        open(json_file_name, 'w').write(json_string)
        self.model.save_weights(h5_file_name, overwrite=True)

    @staticmethod
    def load(file_name, model_type, text_preprocessor, history_length):
        json_file_name, h5_file_name = TpModel.get_full_file_names(file_name)
        model = model_from_json(open(json_file_name, 'r').read())
        model.load_weights(h5_file_name)
        tp_model = TpModel(model_type, text_preprocessor, history_length=history_length)
        tp_model.model = model
        return tp_model

    def delete_model(self, file_name):
        for full_file_name in TpModel.get_full_file_names(file_name):
            try:
                os.remove(full_file_name)
            except OSError:
                pass


if __name__ == '__main__':
    text_file = '../data/pride.txt'
    tp = TextPreProcessor.TextPreProcessor.create_from_text_file(text_file_name=text_file)
    tp = TpModel(ModelType.FirstLSTMModel, tp, history_length=Constants.PreviousWords)
    result = tp.train_on_text_file(text_file, epochs=50)
    print(result)
    tp.save('firstlstm_pride_50epoch')
