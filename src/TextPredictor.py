from __future__ import print_function

import numpy as np

from SequenceModel import SequenceModel
from SequenceProcessor import SequenceProcessor
from Word2Vec import Word2Vec


class TextPredictor:
    def __init__(self, model, sequence_processor):
        """

        :type model: SequenceModel
        :type sequence_processor: SequenceProcessor
        """
        self.sequence_model = model
        self.sequence_processor = sequence_processor

    def get_reply_for_single_query(self, user_text):
        matrix = self.sequence_processor.line_to_matrix(user_text)
        x_in = np.stack([matrix])
        reply_vector = self.sequence_model.predict(x_in)
        return self.sequence_processor.matrix_to_line(reply_vector)

    def get_reply_from_history(self, past_converation):
        tensor = self.sequence_processor.conversation_to_tensor(past_converation)
        reply_vector = self.sequence_model.predict(tensor)
        return self.sequence_processor.matrix_to_line(reply_vector)


if __name__ == '__main__':
    sequence_model = SequenceModel.load('../models/movie_lines_10k')
    sp = SequenceProcessor(Word2Vec(), words_in_sentence=40)
    tp = TextPredictor(model=sequence_model, sequence_processor=sp)
    print(tp.get_reply_for_single_query('who are you'))
