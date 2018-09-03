"""
An agent plays a role like a brain of a solver.
It can act and react 
"""
import logging
import numpy as np
from collections import deque
from keras.models import load_model

from .model import DQNMaker

__all__ = ['BasicAgent', 'DQNAgent']


class Memory(object):
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return str(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[v] for v in idx]


class BasicAgent(object):
    def __init__(self, memory_size=10000):
        self.action_space = action_space

    def initialize(self, state):
        pass

    def pretraining_act(self, state):
        """
        Generate action in the stage of populating memory 
        (initializing data for training).
        """
        return self.action_space.sample()

    def training_act(self, state, step):
        """
        Generate action in the stage of training model.
        """
        return self.action_space.sample()

    def testing_act(self, state):
        """
        Generate action in the stage of testing model.
        """
        return self.action_space.sample()

    def pretraining_react(self, state, reward):
        """
        Reaction of agent when receiving observation in the stage of pre-training.
        """
        pass

    def training_react(self, state, reward):
        """
        Reaction of agent when receiving observation in the stage of training model.
        """
        pass

    def testing_react(self, state, reward):
        """
        Reaction of agent when receiving observation in the stage of testing model.
        """
        pass


class DQNAgent(BasicAgent):
    def __init__(self, action_space, memory_size=10000, 
                 explore_start=1.0, explore_stop=0.01, decay_rate=0.0001, 
                 batch_size=32, gamma=0.99):
        self.model = None
        self.memory = Memory(max_size=memory_size)
        self.action_space = action_space

        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.gamma = gamma

        # cache
        self.state = None
        self.action = None

    def init_model(self, maker, existing_model=None, **kwargs):
        if existing_model is not None:
            self.model = load_model(existing_model)
        else:
            self.model = maker.make(**kwargs)

    def initialize(self, state):
        self.state = state.reshape([1, 4])
        self.action = None

    def pretraining_act(self, state):
        self.action = self.action_space.sample()
        return self.action

    def training_act(self, state, step):    # rewrite this, use self.state
        state = np.reshape(state, [1, 4])
        explore_p = (self.explore_stop + 
                     (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*step))
        if explore_p > np.random.rand():
            self.action = self.action_space.sample()
        else:
            Qs = self.model.predict(state)[0]
            self.action = np.argmax(Qs)
        return self.action

    def testing_act(self, state):
        self.action = self.action_space.sample()
        return self.action

    def pretraining_react(self, state, reward):
        # memorize (current_state, action, reward, next_state)
        next_state = state.reshape([1, 4])
        self.memory.add((self.state, self.action, reward, next_state))
        # update cache
        self.state = next_state

    def training_react(self, state, reward):
        next_state = state.reshape([1, 4])
        self.memory.add((self.state, self.action, reward, next_state))
        self.state = next_state

        # replay
        inputs = np.zeros((self.batch_size, 4))
        targets = np.zeros((self.batch_size, 2))
        minibatch = self.memory.sample(self.batch_size)

        for i, (state, action, reward, next_state) in enumerate(minibatch):
            inputs[i:i+1] = state
            target = reward
            if not np.allclose(next_state, 0):
                target_Q = self.model.predict(next_state)[0]
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            targets[i] = self.model.predict(state)
            targets[i][action] = target

        self.model.fit(inputs, targets, epochs=1, verbose=0)


    def testing_react(self, state, reward):
        pass

    def save_model(self, name='dqnagent_model.h5'):
        import os
        fn = os.path.join(os.getcwd(), name)
        self.model.save(fn)
