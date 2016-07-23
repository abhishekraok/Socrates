from src import TextPreProcessor
from src import FirstLSTMModel


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
        print 'Build model...'
        text = TextPreProcessor.get_clean_words_from_file(text_file_name, 10 ** 7)
        if self.text_processor is None:
            self.text_processor = TextPreProcessor.TextPreProcessor.create_from_text_file(text_file_name=text_file_name)
        train_x = self.text_processor.text_to_vector(text)
        y = train_x[1:]
        x = train_x[:-1]
        print 'Shape of X ', x.shape, ' shape of y ', y.shape
        self.model.train(x, y)
        print 'Training done'
        return self.model.evaluate(x, y)
