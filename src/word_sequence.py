import text_preprocessing

def process_data_for_file(filename):
    string = text_preprocessing.get_clean_words_from_file(filename, 10**6)
    X, vocab = text_preprocessing.word_list_to_one_hot(string)
    print 'The vocabulary size is ', len(vocab)
    print X.shape
    y = X[1:]
    X = X[:-1]
    print 'Does shape of X = shape of y? Ans:', X.shape == y.shape
    return X, y

if __name__ == '__main__':
    X,y = process_data_for_file('../data/pride.txt')
