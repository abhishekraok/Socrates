from src.TextPredictor import TextPredictor


class ChatEngine():
    def __init__(self):
        self.finished = False
        self.count = 3
        self.text_predictor = TextPredictor('model.sequential.p', 'dict1.tsv')

    def chat(self, user_text):
        self.count -= 1
        if not self.count:
            self.finished = True
        return user_text
