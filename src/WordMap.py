class WordMap():
    Unknown = 'UNK'

    def __init__(self, dictionary_file_name=None, words_list=None):
        if dictionary_file_name is None and words_list is None:
            raise Exception("Need something to start dictionary")
        if dictionary_file_name is None:
            self.numbers_to_words_dictionary, self.words_to_numbers_dictionary = WordMap.create_dictionary(words_list)
        else:
            self.numbers_to_words_dictionary = self.load_dictionary(dictionary_file_name)
            self.words_to_numbers_dictionary = dict(((j, i) for i, j in self.numbers_to_words_dictionary.iteritems()))

    def numbers_to_words(self, numbers):
        if max(numbers) > len(self.numbers_to_words_dictionary):
            raise Exception("Max of numbers is greater than dictionary length")
        return [self.numbers_to_words_dictionary[i] for i in numbers]

    def words_to_numbers(self, word_list):
        return [self.words_to_numbers_dictionary.get(i, 0) for i in word_list]

    def save_dictionary(self, file_name):
        with open(file_name, 'w') as f:
            f.write('\n'.join(self.numbers_to_words_dictionary.values()))

    @staticmethod
    def create_dictionary(word_list):
        vocab_list = [WordMap.Unknown] + sorted(list(set(word_list)))
        numbers_to_words_dictionary = dict(((i, j) for i, j in enumerate(vocab_list)))
        words_to_numbers_dictionary = dict(((j, i) for i, j in enumerate(vocab_list)))
        return numbers_to_words_dictionary, words_to_numbers_dictionary

    @staticmethod
    def load_dictionary(dictionary_file_name):
        result = {}
        with open(dictionary_file_name, 'r') as f:
            all_text = f.read()
            return dict((((i,j) for i,j in enumerate(all_text.split('\n')))))


if __name__ == '__main__':
    word_list = ['hi', 'how', 'are', 'you']
    message = ['how', 'are', 'are', 'hi']
    wm = WordMap(words_list=word_list)
    numbers = wm.words_to_numbers(message)
    save_file_name = 'test.txt'
    wm.save_dictionary(save_file_name)
    wm2 = WordMap(dictionary_file_name=save_file_name)
    decoded_message = wm2.numbers_to_words(numbers)
    print decoded_message
