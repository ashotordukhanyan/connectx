from collections import namedtuple
SARSA = namedtuple("SARSA", ["state", "action", "reward", "next_state", "terminal"])
