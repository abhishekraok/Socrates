from __future__ import print_function

from keras.models import model_from_json

import TextPreProcessor
from Constants import Constants
from ModelFactory import ModelFactory
import os
from EnumsCollection import ModelType
from TPModel import TpModel


class SequenceModel:
    """
    The offline class to train and test model for sequence to seqeunce learning.
    """

    def __init__(self, sequence_processor, input_length, output_length):
        self.model = ModelFactory.get_model(ModelType.Sequence,
                                            input_shape=(input_length, sequence_processor.vocabulary_size),
                                            nb_classes=sequence_processor.vocabulary_size)
        self.input_length = input_length
        self.output_length = output_length

    def predict(self, x_in):
        self.model.predict(x_in)

    def train(self, x, y, epoch=1):
        print('Training with input shape', x.shape, ' and output shape ', y.shape)
        self.model.fit(x, y, nb_epoch=epoch)

    # def train_on_text_file(self, text_file_name, epochs):
    #     print('Training on ', text_file_name)
    #     text = TextPreProcessor.get_clean_words_from_file(text_file_name, 10 ** 7)
    #     x, y = self.text_processor.word_list_to_tensor(text, history_length=self.history_length)
    #     print('Shape of X ', x.shape, ' shape of y ', y.shape)
    #     self.model.fit(x, y, nb_epoch=epochs)
    #     print('Training done')
    #     return self.model.evaluate(x, y)

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
