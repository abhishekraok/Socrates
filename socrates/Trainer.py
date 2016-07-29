from __future__ import print_function

from Constants import Constants
from ConversationLoader import ConversationLoader
from ModelFactory import ModelType
from SequenceModel import SequenceModel
from SequenceProcessor import SequenceProcessor
from TextPredictor import TextPredictor
from Word2Vec import Word2Vec


class TrainParameters:
    def __init__(self, model_file_name, queries, epochs, conversation_file, model_type, sentence_length):
        self.model_file_name = model_file_name
        self.queries = queries
        self.epochs = epochs
        self.conversation_file = conversation_file
        self.model_type = model_type
        self.sentence_length = sentence_length


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

    @staticmethod
    def get_trainer_and_predictor(params):
        """
        :type params: TrainParameters
        """
        model_file_name = params.model_file_name
        w2v = Word2Vec()
        sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=params.sentence_length)
        if SequenceModel.isfile(model_file_name):
            model = SequenceModel.load(model_file_name)
        else:
            model = SequenceModel(Constants.Word2VecConstant, input_length=params.sentence_length,
                                  model_type=params.model_type)
        trainer = Trainer(model_file_name=model_file_name, sequence_processor=sp, sequence_model=model)
        tp = TextPredictor(model=trainer.sequence_model, sequence_processor=sp)
        return trainer, tp

    def train(self, conversation, params, text_predictor, total_iterations):
        for i in range(total_iterations):
            print('Iteration number ', i, '/', total_iterations)
            self.train_on_conversation(conversation=conversation, epochs=params.epochs)
            queries = params.queries
            try:
                for query in queries:
                    print('You:', query)
                    reply = text_predictor.get_reply_for_single_query(query)
                    print('Bot:', reply)
            except StandardError:
                print('Got some exception')


def train_from_parameters(params):
    conversation_file = params.conversation_file
    conversation = ConversationLoader.load_conversation_file(conversation_file, reverse=False)
    trainer, text_predictor = Trainer.get_trainer_and_predictor(params)
    trainer.train(conversation=conversation, params=params, text_predictor=text_predictor)


def train_dummy():
    conversation_file = '../data/dummy_convo.txt'
    model_file_name = '../models/dummy_model'
    params = TrainParameters(conversation_file=conversation_file, model_file_name=model_file_name,
                             queries=['who are you', 'what do you do', 'what can you teach'],
                             epochs=100, model_type=ModelType.Sequence10Hidden, sentence_length=10)
    train_from_parameters(params)


def train_movie():
    conversation_file = '../data/movie_lines_cleaned_10k.txt'
    lines = ConversationLoader.load_conversation_file(conversation_file, reverse=False)
    model_file_name = '../models/movie_lines_10k_lstm1k_2layer'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=40)
    if SequenceModel.isfile(model_file_name):
        model = SequenceModel.load(model_file_name)
    else:
        model = SequenceModel(Constants.Word2VecConstant, input_length=40, model_type=ModelType.SeqLayer2Dim1k)
    trainer = Trainer(model_file_name=model_file_name, sequence_processor=sp, sequence_model=model)
    tp = TextPredictor(model=trainer.sequence_model, sequence_processor=sp)
    for i in range(500):
        trainer.train_on_conversation(conversation=lines, epochs=1)
        queries = ['who are you', 'how are you', 'what do you want']
        for query in queries:
            print('You:', query)
            reply = tp.get_reply_for_single_query(query)
            print('Bot:', reply)


def train_english_stack():
    conversation_file = '../data/english_stack.txt'
    lines = ConversationLoader.load_conversation_file(conversation_file, reverse=False)
    model_file_name = '../models/english_stack_lstm1k'
    w2v = Word2Vec()
    sp = SequenceProcessor(word2Vec=w2v, words_in_sentence=40)
    if SequenceModel.isfile(model_file_name):
        model = SequenceModel.load(model_file_name)
    else:
        print('Did not find a previous model file')
        model = SequenceModel(Constants.Word2VecConstant, input_length=40, model_type=ModelType.Sequence1k)
    trainer = Trainer(model_file_name=model_file_name, sequence_processor=sp, sequence_model=model)
    tp = TextPredictor(model=trainer.sequence_model, sequence_processor=sp)
    for i in range(500):
        trainer.train_on_conversation(conversation=lines, epochs=1)
        queries = ['when is it okay to end a sentence in a preposition', 'what is the correct plural of octopus',
                   'what do you want']
        try:
            for query in queries:
                print('You:', query)
                reply = tp.get_reply_for_single_query(query)
                print('Bot:', reply)
        except Exception:
            print('Got some exception')


def train_child_talk():
    conversation_file = '../data/child_talk.txt'
    queries = ['who are you', 'what is the capital of japan',
               'do you want milk']
    model_file_name = '../models/simple_created_1k'
    sentence_length = 10
    params = TrainParameters(conversation_file=conversation_file, model_file_name=model_file_name, queries=queries,
                             epochs=100, model_type=ModelType.Sequence1k, sentence_length=sentence_length)
    train_from_parameters(params)


def train_rowling():
    conversation_file = '../data/jkrowloprah.txt'
    queries = ['who are you', 'Thank you', 'But what do you do']
    model_file_name = '../models/jkrowl_200'
    sentence_length = 40
    params = TrainParameters(conversation_file=conversation_file, model_file_name=model_file_name, queries=queries,
                             epochs=10, model_type=ModelType.Sequence200Hidden, sentence_length=sentence_length)
    train_from_parameters(params)


def train_negative():
    conversation_file = '../data/negative.txt'
    queries = ['hi', 'what do you do', 'what do you teach']
    model_file_name = '../models/negative_100'
    sentence_length = 10
    params = TrainParameters(conversation_file=conversation_file, model_file_name=model_file_name, queries=queries,
                             epochs=100, model_type=ModelType.Sequence100Hidden, sentence_length=sentence_length)
    train_from_parameters(params)


if __name__ == '__main__':
    train_dummy()
