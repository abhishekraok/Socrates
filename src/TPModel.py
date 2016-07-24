from __future__ import print_function

import TextPreProcessor
import FirstLSTMModel
from Constants import Constants


class TpModel():
    """
    The offline class to train models, the interface against which the online service calls.
    This is a general text prediction model and specific details are contained in self.model.
    """

    def __init__(self):
        self.model = FirstLSTMModel.FirstLSTMModel()
        self.text_processor = None

    def predict(self, x_in):
        self.model.predict(x_in)

    def train_words(self, text_file_name):
        print('Training on ', text_file_name)
        text = TextPreProcessor.get_clean_words_from_file(text_file_name, 10 ** 7)
        if self.text_processor is None:
            self.text_processor = TextPreProcessor.TextPreProcessor.create_from_text_file(text_file_name=text_file_name)
        x, y = self.text_processor.word_list_to_tensor(text, Constants.PreviousWords)
        print('Shape of X ', x.shape, ' shape of y ', y.shape)
        self.model.train(x, y)
        print('Training done')
        return self.model.evaluate(x, y)

    def save(self, file_name):
        self.model.save(file_name)


if __name__ == '__main__':
    text_file = '../data/small_pride.txt'
    tp = TpModel()
    print(tp.train_words(text_file))
    tp.save('firstlstm_pride.p')
