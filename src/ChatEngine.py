from src.Actions import Actions
from src.ChatHistory import ChatHistoryKeeper
from src.TextPredictor import TextPredictor
from src.Understander import Understander


class ChatEngine():
    def __init__(self):
        self.finished = False
        self.count = 3
        self.text_predictor = TextPredictor('model.sequential.p', 'dict1.tsv')
        self.query_understander = Understander()
        self.chat_history = ChatHistoryKeeper()

    def chat(self, user_text):
        self.count -= 1
        if not self.count:
            self.finished = True
        reply = self.query_to_actions(user_text)
        self.chat_history.append(user_text)
        return user_text

    def query_to_actions(self, query):
        reply = ''
        action_list = self.query_understander.get_actions(query)
        for action_i in action_list:
            if action_i is Actions.predict:
                reply = self.text_predictor.get_reply(query)
        return reply
