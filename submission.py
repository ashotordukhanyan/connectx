from kaggle_environments import make,utils,evaluate, Environment
from collections import defaultdict,deque,namedtuple
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from numpy.random import choice
import numpy as np
import logging

SARSA = namedtuple("SARSA", ["state", "action", "reward", "next_state","terminal"])

class RLAgent:
    def __init__(self, env : Environment, random = np.random.RandomState() ) -> None:
        self.env_ = env
        self.logger_ = logging.getLogger(self.__str__())
        self.model_ = self.build_model(self.env_.configuration.columns, self.env_.configuration.rows)
        self.memory_ = deque(maxlen=1000000)
        self.learning_rate_ = 0.1
        self.epsilon_ = 0.1
        self.random_ = random

    def build_model(self,columns,rows) -> Model:
        model = Sequential()
        model.add(Dense(20, input_dim=columns*rows, activation='relu')) #input = state + categorical actions taken in 1-hot notation
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(columns, activation='linear'))
        model.compile(optimizer='rmsprop',
                      loss='mean_absolute_error',
                      metrics=['accuracy'])
        logging.info(model.summary())
        return model

    def legal_actions(self, board):
        return [c for c in range(self.env_.configuration.columns) if board[c] == 0]
    def act(self, observation: dict, configuration: dict) -> int:
        legal_actions = self.legal_actions(observation.board)
        if self.random_.random() < self.epsilon_ :
            action = self.random_.choice(self.legal_actions(observation.board))
            reason="exploration"
        else :
            qvals = self.predictStateActions(observation)
            legal_qvals = [ qvals[i] for i in legal_actions]
            action = np.argmax(legal_qvals)
            reason="exploitation"
        self.logger_.debug(f"Given {observation} I choose {action} because of {reason}")
        return int(action)

    def memorize(self, state, action, reward, next_state, terminal ) -> None:
        entry = SARSA(state,action,reward,next_state,terminal)
        self.memory_.append(entry)

    def predictStateAction (self, state, action):
        return self.predictStateActions(state)[action]

    def save_model(self, name: str) -> None:
        name = name+".h5" if not name.endswith(".h5") else name
        self.model_.save(name, overwrite=True)

    def restore_model(self,name:str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_ = load_model(name)

    def predictStateActions (self, state):
        return self.model_.predict([state.board])[0]

    def trainStateAction(self, state, qvalues) -> None:
        self.model_.fit([state.board],[qvalues.tolist()], epochs=1,verbose=self.logger_.isEnabledFor(logging.DEBUG))

    def train(self,samples: int) -> None:
        if len(self.memory_) < samples:
            return
        samples = [ self.memory_[i] for i in self.random_.choice(range(len(self.memory_)),samples)]
        for state,action,reward,next_state,terminal in samples :
            reward = 0 if reward is None else reward
            qvals = self.predictStateActions(state)
            qval = qvals[action]
            if terminal:
                next_max_q = 0
            else:
                next_qs = self.predictStateActions(next_state)
                next_max_q = max(next_qs)
            new_qval = (1-self.learning_rate_)*qval + self.learning_rate_*(reward+next_max_q)
            new_qvals = qvals.copy()
            new_qvals[action] = new_qval
            self.trainStateAction(state,new_qvals)

logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(name)s] [%(levelname)s]:%(message)s')

env = make("connectx",debug=True)
env.render(  )
env.reset()

agent = RLAgent(env )
# Play as the first agent against default "random" agent.
trainer = env.train([None, "random"])
outcomes = defaultdict(int)
for episodeNo in range(100):
    obs = trainer.reset()
    agent.train(25)
    done = False
    moves = 0
    cumreward = 0
    while not done:
        moves+=1
        action = agent.act(obs, env.configuration)
        last_obs = obs
        obs, reward, done, info = trainer.step(action)
        agent.memorize(last_obs,action,reward,obs,done)
        if reward is not None:
            cumreward += reward
        if done:
            outcomestr = "draw" if reward is None else "lost" if reward < 0 else "won"
            logging.info(f"Game {episodeNo} {outcomestr} in {moves} moves. Total reward = {cumreward}")
            outcomes[reward] += 1
            if episodeNo > 0 and episodeNo%25 == 0:
                name=f"episode_{episodeNo}.h5"
                logging.info(f"Saved model under {name}")
                agent.save_model(name)

logging.info(outcomes)
