from cpc.env import CartPoleEnv, AdvancedCartPoleEnv
from cpc.agent import DQNAgent
from cpc.model import DQNMaker
from cpc.solver import CartPoleSolver


def main():
    # Create environment
    env = CartPoleEnv.make()
    # Create model maker
    model_maker = DQNMaker()

    # Create agent (brain) for solver
    agent = DQNAgent(env.action_space, memory_size=500)
    agent.init_model(maker=model_maker)

    # Create solver to solve cartpole problem
    solver = CartPoleSolver(env=env, agent=agent)
    solver.run()


if __name__ == '__main__':
    main()
