class WordMap():
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
        return [self.words_to_numbers_dictionary[i] for i in word_list]

    @staticmethod
    def create_dictionary(word_list):
        vocab_list = sorted(list(set(word_list)))
        numbers_to_words_dictionary = dict(((i, j) for i, j in enumerate(vocab_list)))
        words_to_numbers_dictionary = dict(((j, i) for i, j in enumerate(vocab_list)))
        return numbers_to_words_dictionary, words_to_numbers_dictionary

    @staticmethod
    def load_dictionary(dictionary_file_name):
        result = {}
        with open(dictionary_file_name, 'r') as f:
            all_text = f.read()
            [result[i] =
            for i in all_text]
