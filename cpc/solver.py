import numpy as np
from gym import Env as GymEnv
from .agent import BasicAgent, DQNAgent

__all__ = ['BasicSolver', 'CartPoleSolver']


def skip(func=None, flag=True):
    def outter_wrapper(func):
        def wrapper(*args, **kwargs):
            return None if flag else func(*args, **kwargs)
        return wrapper

    if func:
        # used as a pure decorator
        # (first parameter `func` will be automatically passed in)
        # e.g.
        # ```
        # @skip
        # def foo():
        #     print('foo')
        # ```
        return outter_wrapper(func)
    else:
        # used as a decorator with parameter
        # e.g.
        # ```
        # @skip(flag=True)
        # def foo():
        #     print('foo')
        # ```
        return outter_wrapper


class BasicSolver(object):
    def __init__(self, env=None, agent=None, 
                 skip_pretraining=False,
                 skip_training=False,
                 skip_solving=False):
        self.env = env
        self.agent = agent
        try:
            self._check()
        except:
            raise

        # method rebinding (make each stage skippable)
        self.pretrain = skip(flag=skip_pretraining)(self.pretrain)
        self.train = skip(flag=skip_training)(self.train)
        self.solve = skip(flag=skip_solving)(self.solve)

    def _check(self):
        if not isinstance(self.env, GymEnv):
            raise TypeError('`env` should be an instance of `gym.Env`.')
        if not isinstance(self.agent, BasicAgent):
            raise TypeError('`agent` should be an instance of `BasicAgent`.')

    def pretrain(self):
        """
        Pretraining stage. Can be used to prepare training data.
        This method should be implemented in derived class.
        """
        pass

    def train(self):
        """
        Training stage. Can be used to train your model.
        This method should be implemented in derived class.
        """
        pass

    def solve(self):
        """
        Solving stage.
        This method should be implemented in derived class.
        """
        pass

    def run(self):
        self.pretrain()
        self.train()
        self.solve()

    def terminate(self):
        self.env.close()


class CartPoleSolver(BasicSolver):
    def __init__(self, 
                 pretrain_episodes=50,
                 training_episodes=50,
                 solving_episodes=10,
                 max_steps = 200,
                 render_when_pretraining=False,
                 render_when_training=False,
                 render_when_sovling=True,
                 **kwargs):
        super(CartPoleSolver, self).__init__(**kwargs)

        self.pretrain_episodes = pretrain_episodes
        self.training_episodes = training_episodes
        self.solving_episodes = solving_episodes
        self.max_steps = max_steps

        # flags control for rendering
        self.rwp = render_when_pretraining
        self.rwt = render_when_training
        self.rws = render_when_sovling

    def pretrain(self):
        # initialize environment and agent
        state = self.env.reset()
        self.agent.initialize(state)

        for i in range(self.pretrain_episodes):
            if self.rwp:
                self.env.render()

            action = self.agent.pretraining_act(state)
            state, reward, done, info = self.env.step(action)

            if done:
                self.env.reset()
                state = np.zeros(state.shape)   # indicating the end of an episode
            self.agent.pretraining_react(state, reward)
        self.env.close()

    def train(self):
        # initialize environment and agent
        state = self.env.reset()
        self.agent.initialize(state)

        for i in range(self.training_episodes):
            total_reward = 0
            done = False

            while not done:
                if self.rwt:
                    self.env.render()

                action = self.agent.training_act(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    print('Episode: {}'.format(i),
                          'Total reward: {}'.format(total_reward))
                    self.env.reset()
                    state = np.zeros(state.shape)   # indicating the end of an episode
                self.agent.training_react(state, reward)
        self.agent.save_model()
        self.env.close()

    def solve(self):
        state = self.env.reset()
        self.agent.initialize(state)

        for i in range(self.solving_episodes):
            total_reward = 0
            done = False

            while not done:
                if self.rws:
                    self.env.render()

                action = self.agent.solving_act(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    print('Episode: {}'.format(i),
                          'Total reward: {}'.format(total_reward))
                    self.env.reset()
                self.agent.solving_react(state, reward)
        self.env.close()
