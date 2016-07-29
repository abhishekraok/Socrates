from __future__ import print_function

import numpy as np

from SequenceModel import SequenceModel
from SequenceProcessor import SequenceProcessor
from Word2Vec import Word2Vec
from builtins import input


class TextPredictor:
    def __init__(self, model, sequence_processor):
        """
        Basic prediction class. Gives next conversation from past conversation.

        :type model: SequenceModel
        :type sequence_processor: SequenceProcessor
        """
        self.sequence_model = model
        self.sequence_processor = sequence_processor

    def get_reply_for_single_query(self, user_text):
        matrix = self.sequence_processor.line_to_matrix(user_text)
        x_in = np.stack([matrix])
        reply_vector = self.sequence_model.predict(x_in)
        return self.sequence_processor.matrix_to_line(reply_vector[0, :, :])

    def get_reply_from_history(self, past_conversation):
        tensor = self.sequence_processor.conversation_to_tensor(past_conversation)
        reply_vector = self.sequence_model.predict(tensor)
        print('shape of reply vector', reply_vector.shape)
        return self.sequence_processor.matrix_to_line(reply_vector[-1, :, :])

    @staticmethod
    def load(model_file):
        sequence_model_loaded = SequenceModel.load(model_file)
        config = sequence_model_loaded.model.get_config()
        repeat_layer = config[2]
        if repeat_layer['class_name'] is not 'RepeatVector':
            raise Exception('Repeat layer not in layer number 2. Please specify words in sentence in code')
        words_in_sentence = repeat_layer['config']['n']
        sequence_processor = SequenceProcessor(Word2Vec(), words_in_sentence=words_in_sentence)
        text_predictor = TextPredictor(model=sequence_model_loaded, sequence_processor=sequence_processor)
        return text_predictor


if __name__ == '__main__':
    pre_trained_model = '../model/dummy_model'
    h5_url = "https://1drv.ms/u/s!AizkA_PooBXSkuU1CF09S6EIbW1NGw"
    json_url = "https://1drv.ms/u/s!AizkA_PooBXSkuU0GI_Es3YU6pjRAA"
    SequenceModel.download_from_cloud(pre_trained_model, json_url, h5_url)
    tp = TextPredictor.load(pre_trained_model)
    for i in range(10000):
        input_query = input('User:')
        if input_query in ['quit', 'exit', 'bye']:
            print('Bot: Good bye')
            break
        reply = tp.get_reply_for_single_query(input_query)
        print('Bot:', reply)
