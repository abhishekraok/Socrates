from socrates.EnumsCollection import Actions


class Understander:
    """
    Understands user's queries and returns recommended actions for the given query.
    """

    def __init__(self):
        pass

    def get_actions(self, query):
        return [Actions.predict]
