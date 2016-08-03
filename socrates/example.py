from Socrates.EnumsCollection import ModelType
from Socrates.Socrates import Socrates
from socrates.Parameters import SimpleTrainParameters

if __name__ == '__main__':
    # parameters
    pre_trained_word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_dimension = 300
    model_save_file_name = 'created_model'
    words_in_sentence = 20
    model_type = ModelType.seq2seq_1layer_10hidden_nodes
    conversation_file = '../data/dummy_convo.txt'

    # initializations
    socrates = Socrates.create(pre_trained_word2vec_file=pre_trained_word2vec_file,
                               words_in_sentence=words_in_sentence,
                               word2vec_dimension=word2vec_dimension,
                               model_type=model_type,
                               model_file_name=model_save_file_name)

    # train
    train_params = SimpleTrainParameters(queries=['who are you', 'what do you do'],
                                         epochs=10,
                                         conversation_file=conversation_file,
                                         sentence_length=words_in_sentence,
                                         total_iterations=100)
    socrates.train_using_params(train_params)
