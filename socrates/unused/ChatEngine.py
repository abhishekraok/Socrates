from __future__ import print_function

from socrates.EnumsCollection import Actions
from socrates.TextPredictor import TextPredictor, DummyTextPredictor
from socrates.unused.ChatHistory import ChatHistoryKeeper
from socrates.unused.Understander import Understander


class ChatEngine():
    def __init__(self, model_file_name, dictionary_file_name):
        self.finished = False
        self.count = 3
        if model_file_name and dictionary_file_name:
            self.text_predictor = TextPredictor('model.sequential.p', 'dict1.tsv')
        else:
            self.text_predictor = DummyTextPredictor(None, None)
        self.query_understander = Understander()
        self.chat_history = ChatHistoryKeeper()

    def chat(self, user_text):
        self.count -= 1
        if not self.count:
            self.finished = True
        reply = self.query_to_actions(user_text)
        self.chat_history.append(user_text)
        return reply

    def query_to_actions(self, query):
        reply = ''
        action_list = self.query_understander.get_actions(query)
        for action_i in action_list:
            if action_i is Actions.predict:
                reply = self.text_predictor.get_reply_for_single_query(query)
        return reply


if __name__ == '__main__':
    ce = ChatEngine(None, '../data/MostCommon2266.txt')
    print(ce.chat('hello'))
