"""
Enviroment of problem to be solved.
"""
import gym

__all__ = ['CartPoleEnv', 'AdvancedCartPoleEnv']


class CartPoleEnv(object):
    @staticmethod
    def make():
        return gym.make('CartPole-v0')


class AdvancedCartPoleEnv(object):
    """
    Providing continous action_space.
    """
    @staticmethod
    def make():
        raise NotImplementedError
