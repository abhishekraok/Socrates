from EnumsCollection import ModelType
from TextPredictor import TextPredictor
from Trainer import TrainParameters
from Trainer import Trainer


class Socrates:
    def __init__(self, trainer, text_predictor):
        """
        Creates the chatbot

        :type trainer: Trainer
        :type text_predictor: TextPredictor
        """
        self.trainer, self.text_predictor = trainer, text_predictor

    @staticmethod
    def create_from_params(params):
        """
        :type params: TrainParameters
        """
        text_predictor = TextPredictor.load(params.model_file_name)
        trainer = Trainer(params.model_file_name, sequence_processor=text_predictor.sequence_processor,
                          sequence_model=text_predictor.sequence_model)
        return Socrates(trainer=trainer, text_predictor=text_predictor)

    @staticmethod
    def load(model_file):
        text_predictor = TextPredictor.load(model_file)
        trainer = Trainer(model_file, sequence_processor=text_predictor.sequence_processor,
                          sequence_model=text_predictor.sequence_model)
        return Socrates(trainer, text_predictor=text_predictor)

    def get_reply(self, user_text):
        """
        Replies to a given sentence
        :param user_text: a string of single line conversation
        :return: a string, bot's reply
        """
        return self.text_predictor.get_reply_for_single_query(user_text)

    def train(self, conversation, params, total_iterations):
        return self.trainer.train(conversation=conversation, params=params, text_predictor=self.text_predictor,
                                  total_iterations=total_iterations)

    def save(self, file_name):
        self.text_predictor.sequence_model.save(file_name=file_name)


if __name__ == '__main__':
    train_params = TrainParameters(conversation_file='../data/dummy_convo.txt', model_file_name='../models/dummy_model',
                                   queries=['who are you', 'what do you do', 'what can you teach'],
                                   epochs=100, model_type=ModelType.Sequence10Hidden, sentence_length=10)
    socrates = Socrates.create_from_params(train_params)
    with open(train_params.conversation_file, 'r') as f:
        conversation_from_file = f.read()
        socrates.train(conversation=conversation_from_file, params=train_params, total_iterations=10)
