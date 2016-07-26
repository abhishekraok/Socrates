from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop

from EnumsCollection import ModelType
from Constants import Constants
import seq2seq
from seq2seq.models import SimpleSeq2seq


class ModelFactory(object):
    @staticmethod
    def get_model(model_type, input_shape, nb_classes, output_length=None):
        """
        Gets a model from the Model factory

        :rtype: Sequential
        """
        if model_type.value == ModelType.FirstLSTMModel.value:
            return ModelFactory.get_first_lstm_model(input_shape, nb_classes=nb_classes)
        if model_type.value == ModelType.SimplestModel.value:
            return ModelFactory.get_simplest_model()
        if model_type.value == ModelType.Sequence.value:
            return ModelFactory.get_sequence_model(input_shape=input_shape, nb_classes=nb_classes,
                                                   output_length=output_length)
        if model_type.value == ModelType.SequenceBitAdvanced.value:
            return ModelFactory.get_bit_advanced_sequence_model(input_shape=input_shape, nb_classes=nb_classes,
                                                                output_length=output_length)
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

    @staticmethod
    def get_sequence_model(input_shape, nb_classes, output_length):
        if not output_length:
            raise Exception('Output Length required for sequence model')
        word2vec_dimension = input_shape[1]
        model = SimpleSeq2seq(input_dim=word2vec_dimension, hidden_dim=10, output_length=output_length,
                              output_dim=nb_classes)
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    @staticmethod
    def get_bit_advanced_sequence_model(input_shape, nb_classes, output_length):
        if not output_length:
            raise Exception('Output Length required for sequence model')
        word2vec_dimension = input_shape[1]
        model = SimpleSeq2seq(input_dim=word2vec_dimension, hidden_dim=100, output_length=output_length,
                              output_dim=nb_classes)
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model
