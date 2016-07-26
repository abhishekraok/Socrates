from __future__ import print_function

import os

from EnumsCollection import ModelType
from SequenceModel import SequenceModel
from TPModel import TpModel
from Constants import Constants
from SequenceProcessor import SequenceProcessor
from Word2Vec import Word2Vec


class TextPredictor:
    def __init__(self, model_file_name, sequence_processor, input_length, output_length):
        """

        :type sequence_processor: SequenceProcessor
        """
        if model_file_name:
            self.sequence_model = SequenceModel.load(model_file_name)
        else:
            self.sequence_model = SequenceModel(vector_dimension=Constants.Word2VecConstant,
                                                input_length=input_length, output_length=output_length)
        self.sequence_processor = sequence_processor

    def get_reply(self, user_text):
        x_in = self.sequence_processor.line_to_matrix(user_text)
        reply_vector = self.sequence_model.predict(x_in)
        return self.sequence_processor.matrix_to_line(reply_vector)

    def train_on_conversation_file(self, conversation_file):
        text = self.sequence_processor.file_to_tensor(conversation_file)
        x_in = self.sequence_processor.line_to_matrix(text)
        y = x_in[1:, :, :]  # y is the next line of the conversation
        x = x_in[:-1, :, :]  # remove last line to make shape of x = shape of y
        self.sequence_model.train(x=x, y=y, epoch=1)


if __name__ == '__main__':
    conversation_file = '../data/dummy_convo.txt'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=20)
    tp = TextPredictor(model_file_name=None, sequence_processor=sp, input_length=20, output_length=20)
    tp.train_on_conversation_file(conversation_file)
