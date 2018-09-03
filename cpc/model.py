from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

__all__ = ['ModelMaker']


class ModelMaker(object):
    def __init__(self, **kwargs):
        pass

    def make(self):
        raise NotImplementedError


class DQNMaker(ModelMaker):
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 4)
        self.output_dim = kwargs.get('output_dim', 2)
        self.hidden_size = kwargs.get('hidden_size', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.001)

    def make(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, activation='relu', input_dim=self.input_dim))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dense(self.output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
