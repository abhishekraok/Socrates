from unittest import TestCase
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from numpy import linalg as LA
from sklearn.metrics import f1_score
from WordMap import WordMap
from TextPreprocessor import TextPreProcessor


class TestTextPreprocessor(TestCase):
    def test_text_preprocess(self):
        word_list = ['there', 'here', 'are', 'you', 'hi']
        wm = WordMap(words_list=word_list)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        tp = TextPreProcessor(save_file_name)
        vector = tp.text_to_vector('here are are hi woohoo')
        decoded_message = tp.vector_to_words(vector)
        self.assertEqual(decoded_message, ['here', 'are', 'are', 'hi', TextPreProcessor.Unknown])

    def test_one_hot(self):
        numbers = [3, 1, 2, 0]
        X = TextPreProcessor.numbers_to_one_hot(numbers)
        self.assertEqual((4, 4), X.shape)
        self.assertEqual(X[0, 3], 1)
        self.assertEqual(sum(X[0, :2]), 0)

    def test_one_hot_decode(self):
        numbers = [3, 1, 2, 0]
        X = TextPreProcessor.numbers_to_one_hot(numbers)
        decode = TextPreProcessor.one_hot_to_numbers(X)
        self.assertEqual(decode, numbers)

    def test_encode_decode_words_to_one_hot(self):
        word_list = ['aha', 'bat', 'cat']
        wm = WordMap(words_list=word_list)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        words_list = ['aha', 'bat', 'cat', 'aha']
        tp = TextPreProcessor(save_file_name)
        one_hot = tp.word_list_to_one_hot(words_list)
        decode = tp.one_hot_to_word_list(one_hot)
        self.assertEqual(decode, words_list)


class TestWordMap(TestCase):
    def test_word_map_save_load(self):
        word_list = ['hi', 'how', 'are', 'you']
        message = ['how', 'are', 'are', 'hi']
        wm = WordMap(words_list=word_list)
        numbers = wm.words_to_numbers(message)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        wm2 = WordMap(dictionary_file_name=save_file_name)
        decoded_message = wm2.numbers_to_words(numbers)
        self.assertEqual(decoded_message, message)

    def test_unkown(self):
        word_list = ['hi', 'how', 'are', 'you']
        message = ['how', 'monkey', 'are', 'hi']
        wm = WordMap(words_list=word_list)
        numbers = wm.words_to_numbers(message)
        decoded_message = wm.numbers_to_words(numbers)
        self.assertEqual(decoded_message, ['how', WordMap.Unknown, 'are', 'hi'])
        # def test_words_to_nums(self):
        #     words_list = ['aha', 'bat', 'cat', 'aha']
        #     nums, vocab = text_preprocessing.words_to_numbers(words_list)
        #     self.assertEqual([0, 1, 2, 0], nums)
        #
        # def test_nums_to_words(self):
        #     words_list = ['aha', 'bat', 'cat', 'aha']
        #     nums, vocab = text_preprocessing.words_to_numbers(words_list)
        #     decoded_words = text_preprocessing.numbers_to_words(nums, vocab)
        #     self.assertEqual(decoded_words, words_list)


# class TestText(TestCase):
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
        model.fit(data, labels, nb_epoch=1, batch_size=32)
        predicted = model.predict(data)
        f1 = f1_score(labels, predicted)
        self.assertGreater(f1, 0.7)
