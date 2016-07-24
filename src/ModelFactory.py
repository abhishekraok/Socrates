from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

from src.EnumsCollection import ModelType
from src.Constants import Constants


class ModelFactory(object):
    @staticmethod
    def get_model(model_type):
        if model_type == ModelType.FirstLSTMModel:
            return ModelFactory.get_first_lstm_model()
        if model_type == ModelType.SimplestModel:
            return ModelFactory.get_simplest_model()
        raise Exception("Model type not understood " + str(model_type))

    @staticmethod
    def get_first_lstm_model():
        model = Sequential()
        model.add(LSTM(128, input_dim=Constants.MaxVocabulary))
        model.add(Dense(Constants.MaxVocabulary))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def get_simplest_model():
        model = Sequential()
        model.add(Dense(1, input_dim=1))
        return model
