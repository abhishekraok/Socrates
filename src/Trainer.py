from __future__ import print_function

from Constants import Constants
from ConversationLoader import ConversationLoader
from SequenceModel import SequenceModel
from SequenceProcessor import SequenceProcessor
from TextPredictor import TextPredictor
from Word2Vec import Word2Vec
from ModelFactory import ModelType


class Trainer:
    """
    Responsible for training models offline.
    """

    def __init__(self, model_file_name, sequence_processor, sequence_model):
        """

        :type sequence_processor: SequenceProcessor
        """
        self.sequence_model = sequence_model
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

    def train_on_conversation(self, conversation, epochs=1):
        x_in = self.sequence_processor.conversation_to_tensor(conversation)
        y = x_in[1:, :, :]  # y is the next line of the conversation
        x = x_in[:-1, :, :]  # remove last line to make shape of x = shape of y
        self.sequence_model.train(x=x, y=y, epoch=epochs)
        self.sequence_model.save(self.save_file_name)


def train_dummy():
    conversation_file = '../data/dummy_convo.txt'
    model_file_name = '../models/dummy_model'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=20)
    model = SequenceModel(Constants.Word2VecConstant, input_length=20)
    tp = Trainer(model_file_name=model_file_name, sequence_processor=sp, sequence_model=model)
    tp.train_on_conversation_file(conversation_file)


def train_movie():
    conversation_file = '../data/movie_lines_cleaned_10k.txt'
    lines = ConversationLoader.load_conversation_file(conversation_file, reverse=False)
    model_file_name = '../models/movie_lines_10k_lstm1k'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=40)
    model = SequenceModel(Constants.Word2VecConstant, input_length=40, model_type=ModelType.SeqLayer2Dim1k)
    trainer = Trainer(model_file_name=model_file_name, sequence_processor=sp, sequence_model=model)
    tp = TextPredictor(model=trainer.sequence_model, sequence_processor=sp)
    for i in range(500):
        trainer.train_on_conversation(conversation=lines, epochs=1)
        queries = ['who are you', 'how are you', 'what do you want']
        for query in queries:
            reply = tp.get_reply_for_single_query(query)
            print('Bot:', reply)


if __name__ == '__main__':
    train_movie()
