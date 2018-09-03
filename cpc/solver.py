import numpy as np
from .agent import DQNAgent

__all__ = ['CartPoleSolver']


class CartPoleSolver(object):
    def __init__(self, env=None, agent=None, 
                 populating_episodes=50,
                 training_episodes=50,
                 testing_episodes=10,
                 max_steps = 200,
                 render_when_populating=False,
                 render_when_training=False,
                 render_when_testing=True):
        self.env = env
        self.agent = agent
        self.populating_episodes = agent.batch_size  # rewrite this
        self.training_episodes = training_episodes
        self.testing_episodes = testing_episodes
        self.max_steps = max_steps

        # flags control for rendering
        self.rwp = render_when_populating
        self.rwtr = render_when_training
        self.rwte = render_when_testing

    def pretrain(self):
        # initialize environment and agent
        state = self.env.reset()
        self.agent.initialize(state)

        for i in range(self.populating_episodes):
            if self.rwp:
                self.env.render()

            action = self.agent.pretraining_act(state)
            state, reward, done, info = self.env.step(action)

            if done:
                self.env.reset()
                state = np.zeros(state.shape)   # indicating the end of an episode
            self.agent.pretraining_react(state, reward)

    def train_model(self):
        # initialize environment and agent
        state = self.env.reset()
        self.agent.initialize(state)
        step = 0

        for i in range(self.training_episodes):
            total_reward = 0
            done = False

            while not done:
                step += 1
                if self.rwtr:
                    self.env.render()

                action = self.agent.training_act(state, step)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    print('Episode: {}'.format(i),
                          'Total reward: {}'.format(total_reward))
                    self.env.reset()
                    state = np.zeros(state.shape)   # indicating the end of an episode
                self.agent.training_react(state, reward)
        self.agent.save_model()


    def test_model(self):
        state = self.env.reset()
        self.agent.initialize(state)

        for i in range(self.testing_episodes):
            total_reward = 0
            done = False

            while not done:
                if self.rwte:
                    self.env.render()

                action = self.agent.testing_act(state)
                state, reward, done, info = self.env.step(action)
                total_reward += reward

                if done:
                    print('Episode: {}'.format(i),
                          'Total reward: {}'.format(total_reward))
                    self.env.reset()
                self.agent.testing_react(state, reward)

    def run(self):
        self.pretrain()
        self.train_model()
        self.test_model()

    def terminate(self):
        self.env.close()
