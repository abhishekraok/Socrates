class TrainParameters:
    def __init__(self, model_file_name, queries, epochs, conversation_file, model_type, sentence_length):
        self.model_file_name = model_file_name
        self.queries = queries
        self.epochs = epochs
        self.conversation_file = conversation_file
        self.model_type = model_type
        self.sentence_length = sentence_length


class SimpleTrainParameters:
    def __init__(self, queries, epochs, conversation_file, sentence_length):
        self.queries = queries
        self.epochs = epochs
        self.conversation_file = conversation_file
        self.sentence_length = sentence_length
