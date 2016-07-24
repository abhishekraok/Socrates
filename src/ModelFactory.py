from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

from EnumsCollection import ModelType
from Constants import Constants


class ModelFactory(object):
    @staticmethod
    def get_model(model_type, input_shape, nb_classes):
        if model_type.value == ModelType.FirstLSTMModel.value:
            return ModelFactory.get_first_lstm_model(input_shape, nb_classes=nb_classes)
        if model_type.value == ModelType.SimplestModel.value:
            return ModelFactory.get_simplest_model()
        raise Exception("Model type not understood " + str(model_type))

    @staticmethod
    def get_first_lstm_model(input_shape, nb_classes):
        print('Getting First LSTM model of shape ', input_shape, ' output classes ', nb_classes)
        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def get_simplest_model():
        model = Sequential()
        model.add(Dense(1, input_dim=1))
        return model
