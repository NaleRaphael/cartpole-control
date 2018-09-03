import gym

__all__ = ['CartPoleEnv', 'AdvancedCartPoleEnv']


class CartPoleEnv(object):
    @staticmethod
    def make():
        return gym.make('CartPole-v0')


class AdvancedCartPoleEnv(object):
    @staticmethod
    def make():
        raise NotImplementedError
