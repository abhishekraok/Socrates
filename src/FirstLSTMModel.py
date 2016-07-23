from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import cPickle
from src.Constants import Constants

class FirstLSTMModel(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(Constants.PreviousWords,Constants.MaxVocabulary)))
        self.model.add(Dense(Constants.MaxVocabulary))
        self.model.add(Activation('softmax'))
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def predict(self, x_in):
        return self.model.predict(x_in)

    def train(self, x, y):
        self.model.fit(x, y)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def save(self, file_name):
        cPickle.dumps(self.model)
