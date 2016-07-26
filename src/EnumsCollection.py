from enum import Enum


class Actions(Enum):
    predict = 0
    search_online = 1


class ModelType(Enum):
    FirstLSTMModel = 0
    SimplestModel = 1
    Sequence = 2
    SequenceBitAdvanced = 3
    Sequence10k = 4
    SeqLayer2Dim1k = 5
