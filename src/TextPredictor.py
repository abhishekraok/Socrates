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
        self.save_file_name = model_file_name

    def get_reply(self, user_text):
        x_in = self.sequence_processor.line_to_matrix(user_text)
        reply_vector = self.sequence_model.predict(x_in)
        return self.sequence_processor.matrix_to_line(reply_vector)
