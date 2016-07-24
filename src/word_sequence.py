import TextPreprocessor
from seq2seq.models import SimpleSeq2seq
from keras.models import Sequential
from keras.layers import Dense, Activation
import os
import cPickle


def process_data_for_file(filename):
    string = TextPreprocessor.get_clean_words_from_file(filename, 10 ** 6)
    X, vocab = TextPreprocessor.word_list_to_tensor(string)
    print 'The vocabulary size is ', len(vocab)
    print X.shape
    y = X[1:]
    X = X[:-1]
    print 'Does shape of X = shape of y? Ans:', X.shape == y.shape
    return X, y, vocab

if __name__ == '__main__':
    model_file_name = 'model.sequential.trained.p'
    X,y, vocab = process_data_for_file('../data/pride.txt')
    input_dimension = X.shape[1]
    if os.path.isfile(model_file_name):
        model = cPickle.load(open(model_file_name, 'rb'))
    else:
        model = Sequential()
    model.add(Dense(output_dim=input_dimension, input_dim=input_dimension, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X,y, nb_epoch=1, batch_size=32)
    cPickle.dump(model, open(model_file_name, 'wb'))
    yp = model.predict(X[:100,:])
    words_predicted = TextPreprocessor.one_hot_to_word_list(yp, vocab)
    print words_predicted
