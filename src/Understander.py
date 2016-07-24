from src.EnumsCollection import Actions


class Understander(object):
    def __init__(self):
        pass

    def get_actions(self, query):
        return [Actions.predict]