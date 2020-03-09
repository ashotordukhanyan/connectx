import logging
from kaggle_environments import make, Environment
#from .stateagemt import RLState
from rl.stateagent import RLStateAgent

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s]:%(message)s')
    env = make("connectx", debug=True)
    env.reset()
    agent = RLStateAgent(env, epsilon=0.1, mark=1, model_file="gen_3_mark_1.h5")
    agent.run_training_session(episodes=1000,versus="negamax", output_file="gen3vsnegamax.csv")