import nltk
import numpy as np


def get_clean_words_from_file(file, max_input_length):
    with open(file) as opened_file:
        text = opened_file.read()[:max_input_length]
        return nltk.word_tokenize(clean_text(text))


def clean_text(text, max_input_length=10 ** 10000):
    text = text.replace('\n', ' ')
    max_length = min(max_input_length, len(text))
    rotation = np.random.randint(low=0, high=max_length, size=1)
    text = text[rotation:max_length] + text[:rotation]
    # make sure to remove # for category separation
    text = ''.join(e for e in text if e.isalnum() or e in '.?", ')
    return text


def words_to_numbers(word_list):
    vocab_list = sorted(list(set(word_list)))
    words_to_numbers_dictionary = dict(((j,i) for i,j in enumerate(vocab_list)))
    numbers_to_words_dictionary = dict(((i,j) for i,j in enumerate(vocab_list)))
    return [words_to_numbers_dictionary[i] for i in word_list], numbers_to_words_dictionary


def numbers_to_words(numbers, num2word_dict):
    if max(numbers) > len(num2word_dict):
        raise Exception
    return [num2word_dict[i] for i in numbers]


def numbers_to_one_hot(number_list):
    max_value = max(number_list)
    x = np.zeros([len(number_list), max_value + 1])
    for row, number in enumerate(number_list):
        x[row, number] = 1
    return x


def one_hot_to_numbers(X):
    return list(np.argmax(X, axis=1))


def word_list_to_one_hot(words_list):
    print 'Converting words to one hot representation'
    numbers, vocab = words_to_numbers(words_list)
    return numbers_to_one_hot(numbers), vocab


def one_hot_to_word_list(one_hot_matrix, vocab):
    numbers = one_hot_to_numbers(one_hot_matrix)
    return numbers_to_words(numbers, vocab)


if __name__ == '__main__':
    string = get_clean_words_from_file('../data/pride.txt', 500)
    print string
    vecs, vocab = word_list_to_one_hot(string)
    print vecs
