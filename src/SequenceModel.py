from __future__ import print_function

import os

from keras.models import model_from_json

from EnumsCollection import ModelType
from ModelFactory import ModelFactory
from TPModel import TpModel


class SequenceModel:
    """
    The offline class to train and test model for sequence to sequence learning.
    """

    def __init__(self, vector_dimension=None, input_length=None, output_length=None, model=None):
        if model:
            self.model = model
        else:
            if not (vector_dimension and input_length and output_length):
                raise Exception('Need to either provide model or specify shape')
            self.model = ModelFactory.get_model(ModelType.SequenceBitAdvanced,
                                                input_shape=(input_length, vector_dimension),
                                                nb_classes=vector_dimension, output_length=output_length)

    def predict(self, x_in):
        if len(x_in.shape) is not 3:
            raise Exception('Predict needs dimension 3 input')
        self.model.predict(x_in)

    def train(self, x, y, epoch=1):
        print('Training with input shape', x.shape, ' and output shape ', y.shape)
        self.model.fit(x, y, nb_epoch=epoch)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    @staticmethod
    def get_full_file_names(file_name):
        return file_name + '.json', file_name + '.h5'

    def save(self, file_name):
        json_string = self.model.to_json()
        json_file_name, h5_file_name = TpModel.get_full_file_names(file_name)
        open(json_file_name, 'w').write(json_string)
        self.model.save_weights(h5_file_name, overwrite=True)
        print('Saved model to file ', file_name)

    @staticmethod
    def load(file_name):
        json_file_name, h5_file_name = TpModel.get_full_file_names(file_name)
        model = model_from_json(open(json_file_name, 'r').read())
        model.load_weights(h5_file_name)
        return SequenceModel(model=model)

    @staticmethod
    def delete_model(file_name):
        for full_file_name in TpModel.get_full_file_names(file_name):
            try:
                os.remove(full_file_name)
            except OSError:
                pass
