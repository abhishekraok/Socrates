from unittest import TestCase

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from numpy import linalg as LA
from sklearn.metrics import f1_score

from EnumsCollection import ModelType
from TPModel import TpModel
from TextPreProcessor import TextPreProcessor
from WordMap import WordMap
from Word2Vec import Word2Vec

class TestTPModel(TestCase):
    def test_save_load_model(self):
        test_text_file = 'test.txt'
        with open(test_text_file, 'w') as f:
            f.write('hello how are you')
        tp = TextPreProcessor.create_from_text_file(test_text_file)
        tp_model = TpModel(model_type=ModelType.SimplestModel, history_length=1, text_preprocessor=tp)
        test_save = '../models/test_save'
        tp_model.save(test_save)
        new_model = TpModel.load(test_save, ModelType.SimplestModel, history_length=1, text_preprocessor=tp)
        self.assertEqual(new_model.model.get_weights(), tp_model.model.get_weights())
        tp_model.delete_model(test_save)


class TestTextPreprocessor(TestCase):
    def test_text_preprocess_with_UNK(self):
        word_list = ['there', 'here', 'are', 'you', 'hi']
        wm = WordMap(words_list=word_list)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        tp = TextPreProcessor(save_file_name)
        vector, y = tp.text_to_vector('here are are hi woohoo', 1)
        decoded_message = tp.vector_to_words(vector)
        self.assertEqual(decoded_message, ['here', 'are', 'are', 'hi'])

    def test_text_encod_decode(self):
        word_list = ['there', 'here', 'are', 'you', 'hi']
        wm = WordMap(words_list=word_list)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        tp = TextPreProcessor(save_file_name)
        test_message = 'here are are hi you'
        vector, y = tp.text_to_vector(test_message, 1)
        decoded_message = tp.vector_to_text(vector)
        self.assertEqual(decoded_message, 'here are are hi')

    def test_one_hot(self):
        numbers = [3, 1, 2, 0]
        word_list = ['aha', 'bat', 'cat']
        wm = WordMap(words_list=word_list)
        tp = TextPreProcessor(word_map=wm)
        X, y = tp.numbers_to_tensor(numbers, 1)
        self.assertEqual((3, 1, tp.vocabulary_size), X.shape)
        self.assertEqual(X[0, 0, 3], 1)
        self.assertEqual(sum(X[0, 0, :2]), 0)

    def test_one_hot_decode(self):
        numbers = [3, 1, 2, 0]
        word_list = ['aha', 'bat', 'cat']
        wm = WordMap(words_list=word_list)
        tp = TextPreProcessor(word_map=wm)
        X, y = tp.numbers_to_tensor(numbers, 1)
        decode = TextPreProcessor.one_hot_to_numbers(X)
        self.assertEqual(decode, numbers[:-1])

    def test_encode_decode_words_to_one_hot(self):
        word_list = ['aha', 'bat', 'cat']
        wm = WordMap(words_list=word_list)
        save_file_name = 'test.tsv'
        wm.save_dictionary(save_file_name)
        words_list = ['aha', 'bat', 'cat', 'aha']
        tp = TextPreProcessor(save_file_name)
        one_hot, y = tp.word_list_to_tensor(words_list, 1)
        decode = tp.one_hot_to_word_list(one_hot)
        self.assertEqual(decode, words_list[:-1])


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
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        # generate dummy data
        # train the model, iterating on the data in batches
        # of 32 samples
        model.fit(data, labels, nb_epoch=1, batch_size=32)
        predicted = model.predict(data)
        f1 = f1_score(labels, predicted)
        self.assertGreater(f1, 0.7)

class TestWord2Vec(self):
    def test_word_vec(self):
        word="Congratulations"
        model=Word2Vec()
        words=model.get_words(model.get_vector(word))
        derived_word=words[0]
        self.assertEqual(word,derived_word)
        

