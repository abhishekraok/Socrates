from Socrates.EnumsCollection import ModelType
from Socrates.Socrates import Socrates
from socrates.Parameters import TrainParameters

if __name__ == '__main__':
    # parameters
    pre_trained_word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_dimension = 300
    model_file_name = 'created_model'
    words_in_sentence = 20
    model_type = ModelType.Sequence10Hidden
    conversation_file = '../data/dummy_convo.txt'

    # initializations
    socrates = Socrates.create(pre_trained_word2vec_file=pre_trained_word2vec_file, words_in_sentence=words_in_sentence,
                               word2vec_dimension=word2vec_dimension, model_type=model_type,
                               model_file_name=model_file_name)

    # train
    train_params = TrainParameters(model_file_name=model_file_name, queries=['who are you', 'what do you do'],
                                   epochs=10, conversation_file=conversation_file, model_type=model_type,
                                   sentence_length=words_in_sentence)
    socrates.train()
