import logging
from collections import defaultdict, deque, namedtuple
from typing import List

import numpy as np
# noinspection PyProtectedMember
from kaggle_environments import make, Environment
from numpy.random import choice
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from submission import RLAgent, SARSA
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s]:%(message)s')
env = make("connectx", debug=True)
seed =  42
agent1 = RLAgent(env,epsilon=0.1, model_file="model_g3.h5", name="player3", random=np.random.RandomState(seed))
agent2 = RLAgent(env,epsilon=0.1, model_file="model_g9.h5", name="player9", random=np.random.RandomState(seed+1))
agent3 = RLAgent(env,epsilon=1, name="playerFRESH1", random=np.random.RandomState(seed+1))
agent4 = RLAgent(env,epsilon=1, name="playerFRESH2", random=np.random.RandomState(seed+1))
def gen_function(a: RLAgent) :
    def f (obs,cfg):
        return a.act(obs,cfg)
    return f
tally = defaultdict(int)
for i in range(100):
    env.reset()
    steps = env.run(agents=[gen_function(agent1),gen_function(agent2)])
    last_step = steps[-1]
    p1reward = last_step[0]['reward']
    p2reward = last_step[1]['reward']
    result = "P1 WINS" if p1reward>p2reward else "P2 WINS" if p2reward>p1reward else "DRAW"
    tally[result]= tally[result]+1
    logging.info(f"{result} in {len(steps)} moves")
logging.info(f"Result {tally}")