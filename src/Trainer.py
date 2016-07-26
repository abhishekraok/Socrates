from __future__ import print_function

from Constants import Constants
from SequenceModel import SequenceModel
from SequenceProcessor import SequenceProcessor
from Word2Vec import Word2Vec
import os
from TextPredictor import TextPredictor


class Trainer:
    """
    Responsible for training models offline.
    """

    def __init__(self, model_file_name, sequence_processor):
        """

        :type sequence_processor: SequenceProcessor
        """
        input_length = sequence_processor.words_in_sentence
        output_length = sequence_processor.words_in_sentence
        if model_file_name and os.path.isfile(model_file_name):
            self.sequence_model = SequenceModel.load(model_file_name)
        else:
            self.sequence_model = SequenceModel(vector_dimension=Constants.Word2VecConstant,
                                                input_length=input_length, output_length=output_length)
        self.sequence_processor = sequence_processor
        self.save_file_name = model_file_name

    def train_on_conversation_file(self, conversation_file, epochs=1):
        """
        Trains on a file, and saves it to save_file_name, overwrites.

        :param conversation_file: A text file with each line
        :return:
        """
        x_in = self.sequence_processor.file_to_tensor(conversation_file)
        y = x_in[1:, :, :]  # y is the next line of the conversation
        x = x_in[:-1, :, :]  # remove last line to make shape of x = shape of y
        self.sequence_model.train(x=x, y=y, epoch=epochs)
        self.sequence_model.save(self.save_file_name)


def train_dummy():
    conversation_file = '../data/dummy_convo.txt'
    model_file_name = '../models/dummy_model'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=20)
    tp = Trainer(model_file_name=model_file_name, sequence_processor=sp)
    tp.train_on_conversation_file(conversation_file)


def train_movie():
    conversation_file = '../data/movie_lines_cleaned_10k.txt'
    model_file_name = '../models/movie_lines_10k'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=40)
    trainer = Trainer(model_file_name=model_file_name, sequence_processor=sp)
    tp = TextPredictor(model=trainer.sequence_model, sequence_processor=sp)
    for i in range(50):
        trainer.train_on_conversation_file(conversation_file, epochs=1)
        queries = ['who are you', 'how are you', 'what do you want']
        for i in range(10):
            reply = tp.get_reply_from_history(queries)
            print('Bot:', reply)
            queries.append(reply)


if __name__ == '__main__':
    train_movie()
