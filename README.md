# Socrates
Socrates is a chat bot based on Seq2Seq learning. 

## Getting Started
You will need to download word2vec pre trained file from https://code.google.com/archive/p/word2vec/ and place it in the data folder

## Data 
Please place conversation files in data folder. The expected format of conversation file is each conversation is a text file, and in that file each conversation is a separate line. 

e.g. 
hi
hello
how are you
I'm good, thank you

## Learning
The model tries to predict next line from the current line. 

## Model
Words are converted into vectors using word2vec. Each line is converted to a matrix of size (words_in_sentence, word2vec_dimension). If sentence is less than words_in_sentence then it is padded with 'EOL' which indicates end of line. The tensor created per conversation file is of the dimension (samples, words_in_sentence, word2vec_dimension). 
