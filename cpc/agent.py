"""
An agent plays a role like a brain of a solver.
It can `act` and `react` according to the observation of environment.
    `act`: generate action to affect environment
    `react`: literally, agent do things after receiving feedback from environment
"""
import logging
import numpy as np
from collections import deque
from keras.models import load_model
import gym.spaces

from .model import DQNMaker

__all__ = ['BasicAgent', 'DQNAgent', 'SimpleControl_DQNAgent',
           'PIDControlAgent']


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
    def __init__(self, action_space):
        """
        Parameters
        ----------
        action_space : gym.spaces
            Determine the valid action that agent can generate.
        """
        self.action_space = action_space

    def initialize(self, state):
        pass

    def pretraining_act(self, state):
        """
        Generate action in the stage of pre-training.
        (e.g. initializing data for training).
        """
        return self.action_space.sample()

    def training_act(self, state):
        """
        Generate action in the stage of training model.
        """
        return self.action_space.sample()

    def solving_act(self, state):
        """
        Generate action in the stage of testing model.
        """
        return self.action_space.sample()

    def pretraining_react(self, state, reward):
        """
        React when receiving observation in the stage of pre-training.
        """
        pass

    def training_react(self, state, reward):
        """
        React when receiving observation in the stage of training model.
        """
        pass

    def solving_react(self, state, reward):
        """
        React when receiving observation in the stage of testing model.
        """
        pass


class DQNAgent(BasicAgent):
    def __init__(self, action_space, memory_size=10000, 
                 explore_start=1.0, explore_stop=0.01, decay_rate=0.0001, 
                 batch_size=32, gamma=0.99):
        super(DQNAgent, self).__init__(action_space)
        self.model = None
        self.memory = Memory(max_size=memory_size)

        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.gamma = gamma

        # cache
        self.state = None
        self.action = None
        self.step = 0   # a counter used in training stage

    def init_model(self, maker, existing_model=None, **kwargs):
        if existing_model is not None:
            self.model = load_model(existing_model)
        else:
            self.model = maker.make(**kwargs)

    def initialize(self, state):
        self.state = state.reshape([1, 4])
        self.action = None
        self.step = 0

    def pretraining_act(self, state):
        self.action = self.action_space.sample()
        return self.action

    def training_act(self, state):    # rewrite this, use self.state
        state = np.reshape(state, [1, 4])
        explore_p = (self.explore_stop + 
                     (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*self.step))
        if explore_p > np.random.rand():
            self.action = self.action_space.sample()
        else:
            Qs = self.model.predict(state)[0]
            self.action = np.argmax(Qs)
        # update cache
        self.step += 1
        return self.action

    def solving_act(self, state):
        self.action = self.action_space.sample()
        return self.action

    def pretraining_react(self, state, reward):
        # mission: Fill memory with feedback only
        # memorize (current_state, action, reward, next_state)
        next_state = state.reshape([1, 4])
        self.memory.add((self.state, self.action, reward, next_state))
        # update cache
        self.state = next_state

    def training_react(self, state, reward):
        # mission: 
        # 1. Add new feedback into memory
        # 2. Train model
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

    def solving_react(self, state, reward):
        # note: Model is online for testing, so it do nothing here.
        pass

    def save_model(self, name='dqn_agent_model.h5'):
        import os
        fn = os.path.join(os.getcwd(), name)
        self.model.save(fn)


class SimpleControlAgent(BasicAgent):
    def testing_act(self, state):
        """
        NOTE
        ----
        state (observation from cart-pole system):
            Type: Box(4)
            [cart_position, cart_velocity, pole_angle, pole_tip_velocity]
            cart_position: -2.4 ~ 2.4
            cart_velocity: -inf ~ inf
            pole_angle:    -41.8° ~ 41.8°
            tip_velocity:  -inf ~ inf

        actions:
            Type: Discrete(2)
            0: push cart to left
            1: push cart to right

        Decision table:
            angle \ vel |   > 0   |   <= 0
            ---------------------------------
                 > 0    |    1    |    1
            ---------------------------------
                <= 0    |    0    |    0
            * We care about the angle of pole only.
        """
        self.action = 1 if state[2] > 0 else 0
        return self.action


class PIDControlAgent(BasicAgent):
    """
    This agent is a pure PID controller, so its parameters (kp, ki, kd) is not
    going to be tuned automatically.
    However, we can apply a learning model to tune them later.
    """
    def __init__(self, action_space, fs, kp=1.2, ki=1.0, kd=0.001, set_angle=0.0):
        """
        Parameters
        ----------
        action_space : gym.spaces
            Determine the valid action that agent can generate.
        fs : float
            Samping frequency. (Hz)
        kp : float
            Gain of propotional controller.
        ki : float
            Gain of integral controller.
        kd : float
            Gain of derivative controller.
        """
        super(PIDControlAgent, self).__init__(action_space)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.set_angle = set_angle
        self.tau = 1.0/fs

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

        # cache
        self.output = 0.0
        self.err_prev = 0.0

    def update(self, v_in, v_fb):
        """
        Parameters
        ----------
        v_in : int or float
            Input command.
        v_fb : int or float
            Feedback from observer.

        Returns
        -------
        output : float
            Output command.

        Note
        ----
        Output of PID controller:
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        err = v_in - v_fb

        self.p_term = err
        self.i_term += err*self.tau
        self.d_term = (err - self.err_prev)*self.tau
        self.output = self.kp*self.p_term + self.ki*self.i_term + self.kd*self.d_term

        # update cache
        self.err_prev = err

        return self.output

    def choose_action(self, val):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = 0 if val >= 0 else 1
        elif isinstance(self.action_space, gym.spaces.Box):
            action = None   # rewrite this for continous action space
        return action

    def solving_act(self, state):
        output = self.update(self.set_angle, state[2])
        temp = self.choose_action(output)
        self.action = temp
        return self.action
