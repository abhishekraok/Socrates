# Socrates
Socrates is a chat bot based on Seq2Seq learning. 

## Dependencies
You will need to download word2vec pre trained file from https://code.google.com/archive/p/word2vec/ and place it in the data folder.
The following libraries are required

1. Theano
2. Keras
3. [Seq2Seq](https://github.com/farizrahman4u/seq2seq)
4. GenSim

Check the requirements.txt for full list

## Data 
Please place conversation files in data folder. The expected format of conversation file is each conversation is a text file, and in that file each conversation is a separate line. See the sample conversation file in [/data/dummy_convo.txt]( https://github.com/abhishekraok/Socrates/blob/master/data/dummy_convo.txt)

## Model
Words are converted into vectors using word2vec. Each line is converted to a matrix of size (words_in_sentence, word2vec_dimension). If sentence is less than words_in_sentence then it is padded with 'EOL' which indicates end of line. The tensor created per conversation file is of the dimension (samples, words_in_sentence, word2vec_dimension). 

## Learning
The model tries to predict next line from the current line. 

## Unit tests
All the unit tests are in the file [/socrates/Tests.py](https://github.com/abhishekraok/Socrates/blob/master/socrates/Tests.py)
