class ConversationLoader:
    @staticmethod
    def load_conversation_file(file_name, reverse=False):
        with open(file_name, 'r') as f:
            lines = f.readlines()
            if reverse:
                return lines[::-1]
            else:
                return lines
