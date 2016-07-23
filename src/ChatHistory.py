class ChatHistoryKeeper(object):
    def __init__(self):
        self.past_queries = []

    def append(self, query):
        self.past_queries.append(query)