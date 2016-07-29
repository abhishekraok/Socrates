from __future__ import print_function

import os
import shutil

from keras.models import model_from_json
from keras.utils.data_utils import get_file

from ModelFactory import ModelFactory


class SequenceModel:
    """
    The offline class to train and test model for sequence to sequence learning.
    """

    def __init__(self, vector_dimension=None, input_length=None, model=None, model_type=None):
        output_length = input_length
        if model:
            self.model = model
        else:
            print('Creating model of dimension ', input_length, vector_dimension, ' and output len ', output_length)
            if not (vector_dimension and input_length and output_length):
                raise Exception('Need to either provide model or specify shape')
            self.model = ModelFactory.get_model(model_type=model_type,
                                                input_shape=(input_length, vector_dimension),
                                                nb_classes=vector_dimension, output_length=output_length)
        print(self.model.summary())

    def predict(self, x_in):
        if len(x_in.shape) is not 3:
            raise Exception('Predict needs dimension 3 input')
        return self.model.predict(x_in)

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
        json_string = json_string.replace('SimpleSeq2seq', 'Sequential')
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(file_name)
        open(json_file_name, 'w').write(json_string)
        self.model.save_weights(h5_file_name, overwrite=True)
        print('Saved model to file ', file_name)

    @staticmethod
    def get_full_file_names(file_name):
        return (file_name + '.json', file_name + '.h5')

    @staticmethod
    def load(file_name):
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(file_name)
        model = model_from_json(open(json_file_name, 'r').read())
        model.compile(optimizer='rmsprop', loss='mse')
        model.load_weights(h5_file_name)
        print('Loaded file ', file_name)
        return SequenceModel(model=model)

    @staticmethod
    def download_from_cloud(model_file_name, json_url, h5_url):
        print('Downloading from cloud')
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(model_file_name)
        downloaded_json = get_file(os.path.normpath(json_file_name), origin=json_url)
        if downloaded_json != json_file_name:
            shutil.copy(downloaded_json, json_file_name)
        downloaded_h5 = get_file(os.path.normpath(h5_file_name), origin=h5_url)
        if downloaded_h5 != h5_file_name:
            shutil.copy(downloaded_h5, h5_file_name)

    @staticmethod
    def delete_model(file_name):
        for full_file_name in SequenceModel.get_full_file_names(file_name):
            try:
                os.remove(full_file_name)
            except OSError:
                pass

    @staticmethod
    def isfile(model_file_name):
        json_file_name, h5_file_name = SequenceModel.get_full_file_names(model_file_name)
        return os.path.isfile(json_file_name) and os.path.isfile(h5_file_name)
