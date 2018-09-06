from argparse import ArgumentParser

import numpy as np
import matplotlib.pylab as plt

from cpc.env import CartPoleEnv, AdvancedCartPoleEnv
from cpc.agent import BasicAgent, DQNAgent, SimpleControlAgent
from cpc.model import DQNMaker
from cpc.solver import CartPoleSolver


def basic_solver():
    # Create environment
    env = CartPoleEnv.make()

    # Create agent (brain) for solver
    agent = BasicAgent(env.action_space)

    # Create solver to solve cartpole problem
    # NOTE: pretraining and training stage is not required for this solver
    solver = CartPoleSolver(env=env, agent=agent, 
                            skip_pretraining=True,
                            skip_training=True)
    solver.run()


def dqn_solver():
    # Create environment
    env = CartPoleEnv.make()
    # Create model maker
    model_maker = DQNMaker()

    # Create agent (brain) for solver
    agent = DQNAgent(env.action_space, memory_size=500)
    # Assign a DQN model maker to agent to create model
    agent.init_model(maker=model_maker)

    # Create solver to solve cartpole problem
    solver = CartPoleSolver(env=env, agent=agent, 
                            pretrain_episodes=agent.batch_size)
    solver.run()


def simple_control_solver():
    env = CartPoleEnv.make()
    model_maker = DQNMaker()

    agent = SimpleControlAgent(env.action_space)

    # NOTE: pretraining and training stage is not required for this solver
    solver = CartPoleSolver(env=env, agent=agent, 
                            skip_pretraining=True,
                            skip_training=True)
    solver.run()


def pid_control_solver():
    from cpc.agent import PIDControlAgent
    env = CartPoleEnv.make()
    model_maker = DQNMaker()

    # NOTE: kp, ki, kd are tuned manually, they are not the optimal parameter
    # for this PID controller
    agent = PIDControlAgent(env.action_space, 
                            env.metadata['video.frames_per_second'],
                            kp=1, ki=0, kd=75)

    # NOTE: pretraining and training stage is not required for this solver
    solver = CartPoleSolver(env=env, agent=agent, 
                            skip_pretraining=True,
                            skip_training=True)
    solver.run()


solver_entry = {
    'basic': basic_solver,
    'dqn': dqn_solver,
    'simple_control': simple_control_solver,
    'pid_control': pid_control_solver
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--solver', dest='solver', default='basic')
    return parser.parse_args()


def main():
    args = parse_args()
    solver_entry[args.solver]()


if __name__ == '__main__':
    main()
