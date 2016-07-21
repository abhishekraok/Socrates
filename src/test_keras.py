from unittest import TestCase
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from numpy import linalg as LA
from sklearn.metrics import f1_score
from src import text_preprocessing


class TestText(TestCase):
    def test_words_to_nums(self):
        words_list = ['aha', 'bat', 'cat', 'aha']
        nums = text_preprocessing.words_to_numbers(words_list)
        self.assertEqual([0, 1, 2, 0], nums)

    def test_one_hot(self):
        numbers = [3,1, 2, 0]
        X = text_preprocessing.one_hot_encode(numbers)
        self.assertEqual((4,4), X.shape)
        self.assertEqual(X[0,3], 1)
        self.assertEqual(sum(X[0,:2]),0)

    def test_one_hot_decode(self):
        numbers = [3,1, 2, 0]
        X = text_preprocessing.one_hot_encode(numbers)
        decode = text_preprocessing.one_hot_decode(X)
        self.assertEqual(decode, numbers)

class TestKeras(TestCase):
    def test_model_compiles(self):
        model = Sequential([
            Dense(32, input_dim=784),
            Activation('relu'),
            Dense(10),
            Activation('softmax'),
        ])
        self.assertIsNotNone(model)

    def test_model_train_predict(self):
        input_dimension = 2
        data = np.random.random((1000, input_dimension))
        data_norm = LA.norm(data, axis=1)
        labels = [1 if i > 0.6 else 0 for i in data_norm]
        model = Sequential()
        model.add(Dense(1, input_dim=input_dimension, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        # generate dummy data
        # train the model, iterating on the data in batches
        # of 32 samples
        model.fit(data, labels, nb_epoch=10, batch_size=32)
        predicted = model.predict(data)
        f1 = f1_score(labels, predicted)
        self.assertGreater(f1, 0.7)
