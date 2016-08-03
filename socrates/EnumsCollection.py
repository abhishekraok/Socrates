from enum import Enum


class Actions(Enum):
    predict = 0
    search_online = 1


class ModelType(Enum):
    FirstLSTMModel = 0
    SimplestModel = 1
    seq2seq_1layer_10hidden_nodes = 2
    seq2seq_1layer_100hidden_nodes = 3
    seq2seq_1layer_1000hidden_nodes = 4
    seq2seq_2layer_1000hidden_nodes = 5
    seq2seq_1layer_2000hidden_nodes = 6
