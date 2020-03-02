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

SARSA = namedtuple("SARSA", ["state", "action", "reward", "next_state", "terminal"])

class RLStateAgent:
    def __init__(self, env: Environment, random=np.random.RandomState(), *, learning_rate=0.1, epsilon=0.1,
                 model_file=None) -> None:
        self.env_ = env
        self.logger_ = logging.getLogger(self.__str__())
        if model_file is None:
            self.model_ = self.build_model(self.env_.configuration.columns, self.env_.configuration.rows)
        else:
            self.restore_model(model_file)
        self.memory_ = deque(maxlen=1000000)
        self.learning_rate_ = learning_rate
        self.epsilon_ = epsilon
        self.random_ = random

    @staticmethod
    def build_model(columns, rows) -> Model:
        model = Sequential()
        model.add(Dense(20, input_dim=columns * rows,
                        activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='rmsprop',
                      loss='mean_absolute_error',
                      metrics=['accuracy'])
        logging.info(model.summary())
        return model

    def legal_actions(self, board):
        return [c for c in range(self.env_.configuration.columns) if board[c] == 0]

    def next_board(self,board: List[int], action:int) ->List[int]:
        nb = board.copy()
        pass
        #TODO

    # noinspection PyUnusedLocal
    def act(self, observation: dict, configuration: dict) -> int:
        legal_actions = self.legal_actions(observation['board'])
        if self.random_.random() < self.epsilon_:
            action = self.random_.choice(legal_actions)
            reason = "exploration"
        else:
            qvals = self.predict_state_actions(observation)
            legal_qvals = [qvals[i] for i in legal_actions]
            action = legal_actions[np.argmax(legal_qvals)]
            reason = "exploitation"
        self.logger_.debug(f"Given {observation} I choose {action} because of {reason}")
        return int(action)

    def memorize(self, state, action, reward, next_state, terminal) -> None:
        entry = SARSA(state, action, reward, next_state, terminal)
        self.memory_.append(entry)

    def predict_state_action(self, state, action):
        return self.predict_state_actions(state)[action]

    def save_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_.save(name, overwrite=True)

    def restore_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_ = load_model(name)

    def predict_state_actions(self, state):
        return self.model_.predict([state.board])[0]

    def train_state_action(self, state, qvalues) -> None:
        self.model_.fit([state.board], [qvalues.tolist()], epochs=1, verbose=self.logger_.isEnabledFor(logging.DEBUG))

    def refit_model(self, samples: int) -> None:
        if len(self.memory_) < samples:
            return
        samples = [self.memory_[i] for i in self.random_.choice(range(len(self.memory_)), samples)]
        for state, action, reward, next_state, terminal in samples:
            reward = 0 if reward is None else reward
            qvals = self.predict_state_actions(state)
            qval = qvals[action]
            if terminal:
                next_max_q = 0
            else:
                next_qs = self.predict_state_actions(next_state)
                next_max_q = max(next_qs)
            new_qval = (1 - self.learning_rate_) * qval + self.learning_rate_ * (reward + next_max_q)
            new_qvals = qvals.copy()
            new_qvals[action] = new_qval
            self.train_state_action(state, new_qvals)

    def run_training_session(self, episodes=100, versus="random", model_weights_file_suffix="",
                             output_file=None) -> dict:
        # Play as the first agent against default "random" agent.
        if output_file is not None:
            fp = open(output_file, "w")
            fp.write("Game,Outcome,Moves,CumReward,LastReward\n")
            fp.flush()

        trainer = env.train([versus, None])
        outcomes = defaultdict(int)
        total_reward = 0
        for episodeNo in range(episodes):
            self.env_.reset()
            obs = trainer.reset()
            self.refit_model(50)
            done = False
            moves = 0
            cumreward = 0
            while not done:
                moves += 1
                action = self.act(obs, env.configuration)
                last_obs = obs
                obs, reward, done, info = trainer.step(action)
                self.memorize(last_obs, action, reward, obs, done)
                if done:
                    reward = reward * 10 - 5 # BOOST AND CENTER
                cumreward += reward
                if done:
                    outcomestr = "draw" if reward == 0 else "won" if reward > 0 else "lost"
                    self.logger_.debug(
                        f"Game {episodeNo} {outcomestr} in {moves} moves. Reward = {cumreward} Last reward = {reward}")
                    outcomes[outcomestr] += 1
                    total_reward += cumreward
                    if output_file is not None:
                        fp.write(f"{episodeNo},{outcomestr},{moves},{cumreward},{reward}\n")
                        fp.flush()
        outcomes["total_reward"] =  total_reward
        return outcomes

class RLAgent:
    def __init__(self, env: Environment, random=np.random.RandomState(), *, learning_rate=0.1, epsilon=0.1,
                 model_file=None, name="RLAgent") -> None:
        self.env_ = env
        self.logger_ = logging.getLogger(name)
        if model_file is None:
            self.model_ = self.build_model(self.env_.configuration.columns, self.env_.configuration.rows)
        else:
            self.restore_model(model_file)
        self.memory_ = deque(maxlen=1000000)
        self.learning_rate_ = learning_rate
        self.epsilon_ = epsilon
        self.random_ = random

    @staticmethod
    def build_model(columns, rows) -> Model:
        model = Sequential()
        model.add(Dense(20, input_dim=columns * rows,
                        activation='softmax'))  # input = state + categorical actions taken in 1-hot notation
        model.add(Dense(20, activation='softmax'))
        model.add(Dense(20, activation='softmax'))
        model.add(Dense(columns, activation='linear'))
        model.compile(optimizer='rmsprop',
                      loss='mean_absolute_error',
                      metrics=['accuracy'])
        logging.info(model.summary())
        return model

    def legal_actions(self, board):
        return [c for c in range(self.env_.configuration.columns) if board[c] == 0]

    # noinspection PyUnusedLocal
    def act(self, observation: dict, configuration: dict) -> int:
        legal_actions = self.legal_actions(observation['board'])
        if self.random_.random() < self.epsilon_:
            action = self.random_.choice(legal_actions)
            reason = "exploration"
        else:
            qvals = self.predict_state_actions(observation)
            legal_qvals = [qvals[i] for i in legal_actions]
            self.logger_.debug(f"QVALS ={qvals}")
            self.logger_.debug(f"LEGAL ACTIONS ={legal_actions}")
            action = legal_actions[np.argmax(legal_qvals)]
            reason = "exploitation"
        self.logger_.debug(self.env_.render(mode="ansi"))
        self.logger_.debug(f"Given {observation} I choose {action} because of {reason}")
        return int(action)

    def memorize(self, state, action, reward, next_state, terminal) -> None:
        entry = SARSA(state, action, reward, next_state, terminal)
        self.memory_.append(entry)

    def predict_state_action(self, state, action):
        return self.predict_state_actions(state)[action]

    def save_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_.save(name, overwrite=True)

    def restore_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_ = load_model(name)

    def predict_state_actions(self, state):
        return self.model_.predict([state.board])[0]

    def train_state_action(self, state, qvalues) -> None:
        self.model_.fit([state.board], [qvalues.tolist()], epochs=1, verbose=self.logger_.isEnabledFor(logging.DEBUG))

    def refit_model(self, samples: int) -> None:
        if len(self.memory_) < samples:
            return
        samples = [self.memory_[i] for i in self.random_.choice(range(len(self.memory_)), samples)]
        for state, action, reward, next_state, terminal in samples:
            reward = 0 if reward is None else reward
            qvals = self.predict_state_actions(state)
            qval = qvals[action]
            if terminal:
                next_max_q = 0
            else:
                next_qs = self.predict_state_actions(next_state)
                next_max_q = max(next_qs)
            new_qval = (1 - self.learning_rate_) * qval + self.learning_rate_ * (reward + next_max_q)
            new_qvals = qvals.copy()
            new_qvals[action] = new_qval
            self.train_state_action(state, new_qvals)

    def run_training_session(self, episodes=100, versus="random", model_weights_file_suffix="",
                             output_file=None) -> dict:
        # Play as the first agent against default "random" agent.
        if output_file is not None:
            fp = open(output_file, "w")
            fp.write("Game,Outcome,Moves,CumReward,LastReward\n")
            fp.flush()

        trainer = env.train([versus, None])
        outcomes = defaultdict(int)
        total_reward = 0
        for episodeNo in range(episodes):
            self.env_.reset()
            obs = trainer.reset()
            self.refit_model(50)
            done = False
            moves = 0
            cumreward = 0
            while not done:
                moves += 1
                action = self.act(obs, env.configuration)
                last_obs = obs
                obs, reward, done, info = trainer.step(action)
                self.memorize(last_obs, action, reward, obs, done)
                if done:
                    reward = reward * 10 - 5 # BOOST AND CENTER
                cumreward += reward
                if done:
                    outcomestr = "draw" if reward == 0 else "won" if reward > 0 else "lost"
                    self.logger_.debug(
                        f"Game {episodeNo} {outcomestr} in {moves} moves. Reward = {cumreward} Last reward = {reward}")
                    outcomes[outcomestr] += 1
                    total_reward += cumreward
                    if output_file is not None:
                        fp.write(f"{episodeNo},{outcomestr},{moves},{cumreward},{reward}\n")
                        fp.flush()
        outcomes["total_reward"] =  total_reward
        return outcomes

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s]:%(message)s')
    env = make("connectx", debug=True)

    for generation in range(6,20):
        env.reset()
        agent = RLAgent(env, epsilon=0.2, model_file=f"model_g{generation}.h5")
        prevgen_agent = RLAgent(env,epsilon=0,model_file=f"model_g{generation-1}.h5")

        def prevgen_act(observation, configuration):
            return prevgen_agent.act(observation,configuration)

        outcomes = agent.run_training_session(episodes=500, versus=prevgen_act)
        agent.save_model(f"model_g{generation+1}.h5")
        logging.info(f"Generation {generation} outcomes {outcomes}")
