import cPickle
import os
import TextPreprocessor

from src.WordMap import WordMap


class TextPredictor():
    def __init__(self, model_file_name, word_map_file):
        if os.path.isfile(model_file_name):
            model = cPickle.load(open(model_file_name, 'rb'))
        else:
            raise Exception("model file not found " + model_file_name)
        self.model = model
        self.text_processor = TextPreprocessor.TextPreProcessor(word_map_file)

    def get_reply(self, user_text):
        x_in = self.text_processor.text_to_vector(text=user_text)
        reply_vector = self.model.predict(x_in)
        return self.text_processor.vector_to_text(reply_vector)
