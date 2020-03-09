import logging
from collections import defaultdict, deque
from typing import List
import numpy as np
# noinspection PyProtectedMember
from kaggle_environments import make, Environment
from numpy.random import choice

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.models import load_model
from rl.core import SARSA

class RLStateAgent:
    FIT_EPOCHS=3
    def __init__(self, env: Environment, mark:int, random=np.random.RandomState(), *, learning_rate=0.1, epsilon=0.1,
                 model_file=None) -> None:
        self.env_ = env
        self.logger_ = logging.getLogger(self.__str__())
        if model_file is None:
            self.model_ = self.build_model(self.configuration.columns, self.configuration.rows)
        else:
            self.restore_model(model_file)
        self.memory_ = deque(maxlen=1000000)
        self.learning_rate_ = learning_rate
        self.epsilon_ = epsilon
        self.random_ = random
        self.mark_ = mark

    @property
    def configuration(self):
        return self.env_.configuration

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

    def next_board(self,board: List[int], mark:int, action:int) ->List[int]:
        nb = board.copy()
        row = self._next_free_row(board,action)
        nb[row*self.configuration.columns + action] = mark
        return nb

    def _next_free_row(self, board:List[int], column:int) -> int:
        for r in range(self.configuration.rows-1,-1,-1):
            if board[r*self.configuration.columns + column] == 0 :
                return r
        return None

    def state_qvals(self, boards: List[List[int]]) -> float:
        return [x[0] for x in self.model_.predict(boards)]

    def opponent_mark(self) -> int:
        return 1 if self.mark_ == 2 else 2
    def act(self, observation: dict) -> int:
        legal_actions = self.legal_actions(observation['board'])
        if self.random_.random() < self.epsilon_:
            best_action = self.random_.choice(legal_actions)
            reason = "exploration"
        else:
            min_max_val = -1*np.inf
            best_action = -1
            for index in range(len(legal_actions)):
                action = legal_actions[index]
                next_board = self.next_board(observation.board,self.mark_,action)
                opponent_actions = self.legal_actions(next_board)
                next_next_boards = [ self.next_board(next_board,self.opponent_mark(),a) for a in opponent_actions]
                state_vals = self.state_qvals(next_next_boards)
                self.logger_.debug(f"My action {action} results in possible boards with vals {state_vals}")
                min_state_val = min(state_vals)
                if min_state_val > min_max_val:
                    min_max_val = min_state_val
                    best_action = legal_actions[index]
            reason = "exploitation"
        self.logger_.debug(f"Given {observation} I choose {best_action} because of {reason}")
        return int(best_action)

    def memorize(self, state, action, reward, next_state, terminal) -> None:
        entry = SARSA(state, action, reward, next_state, terminal)
        self.memory_.append(entry)

    def save_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_.save(name, overwrite=True)

    def restore_model(self, name: str) -> None:
        name = name + ".h5" if not name.endswith(".h5") else name
        self.model_ = load_model(name)

    def predict_state_values(self, states : List[List[int]]) -> List[float]:
        return self.model_.predict(states)

    def train_state_action(self, state, qval:float) -> None:
        ##self.model_.fit([state.board], [qval], epochs=RLStateAgent.FIT_EPOCHS, verbose=self.logger_.isEnabledFor(logging.DEBUG))
        self.model_.fit([state.board], [qval], epochs=RLStateAgent.FIT_EPOCHS, verbose=False)

    def refit_model(self, samples: int) -> None:
        if len(self.memory_) < samples:
            return
        samples = [self.memory_[i] for i in self.random_.choice(range(len(self.memory_)), samples)]
        for state, action, reward, next_state, terminal in samples:
            reward = 0 if reward is None else reward
            qval = self.state_qvals([state.board])[0]
            if terminal:
                next_qval=0
            else:
                next_qval = self.state_qvals([next_state.board])[0]
            new_qval = (1 - self.learning_rate_) * qval + self.learning_rate_ * (reward + next_qval)
            self.train_state_action(state, new_qval)

    def run_training_session(self, episodes=100, versus="random", model_weights_file_suffix="",
                             output_file=None) -> dict:
        # Play as the first agent against default "random" agent.
        if output_file is not None:
            fp = open(output_file, "w")
            fp.write("Game,Outcome,Moves,CumReward,LastReward\n")
            fp.flush()

        trainer = self.env_.train([versus, None]) if self.mark_ == 2 else self.env_.train([None, versus])
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
                action = self.act(obs )
                last_obs = obs
                obs, reward, done, info = trainer.step(action)
                self.memorize(last_obs, action, reward, obs, done)
                if done:
                    reward = reward * 10 - 5 # BOOST AND CENTER
                cumreward += reward
                if done:
                    outcomestr = "draw" if reward == 0 else "won" if reward > 0 else "lost"
                    self.logger_.info(
                        f"Game {episodeNo} {outcomestr} in {moves} moves. Reward = {cumreward} Last reward = {reward}")
                    outcomes[outcomestr] += 1
                    total_reward += cumreward
                    if output_file is not None:
                        fp.write(f"{episodeNo},{outcomestr},{moves},{cumreward},{reward}\n")
                        fp.flush()
        outcomes["total_reward"] =  total_reward
        return outcomes
