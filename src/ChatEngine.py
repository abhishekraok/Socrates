class ChatEngine():
    def __init__(self):
        self.finished = False
        self.count = 3

    def chat(self, user_text):
        self.count -= 1
        if not self.count:
            self.finished = True
        return user_text