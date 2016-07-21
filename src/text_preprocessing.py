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
    vocab = sorted(list(set(word_list)))
    return [vocab.index(i) for i in word_list]


if __name__ == '__main__':
    string = get_clean_words_from_file('../data/pride.txt', 500)
    print string
    nums = words_to_numbers(string)
    print nums
