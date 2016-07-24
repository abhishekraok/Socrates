from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import cPickle
from src.Constants import Constants
import numpy as np

class FirstLSTMModel(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(1,Constants.MaxVocabulary)))
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

if __name__ == '__main__':
    model = FirstLSTMModel()
    x = np.random.random((100,1, Constants.MaxVocabulary))
    y = np.random.random((100,Constants.MaxVocabulary))
    model.train(x,y)
    print model.evaluate(x,y)


