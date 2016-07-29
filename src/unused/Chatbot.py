from __future__ import print_function

from src.unused.ChatEngine import ChatEngine


class ChatBot:
    def __init__(self):
        self.chat_engine = ChatEngine(None, None)

    def chat(self):
        while not self.chat_engine.finished:
            user_text = input('User:')
            bot_reply = self.chat_engine.chat(user_text)
            print(bot_reply)


if __name__ == '__main__':
    socrates = ChatBot()
    socrates.chat()