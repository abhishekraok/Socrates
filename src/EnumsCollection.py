from enum import Enum


class Actions(Enum):
    predict = 0
    search_online = 1


class ModelType(Enum):
    FirstLSTMModel = 1
    SimplestModel = 2