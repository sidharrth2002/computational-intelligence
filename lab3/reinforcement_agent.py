
from copy import deepcopy
import logging
import math
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from evolved_nim import BrilliantEvolvedAgent
from fixed_rule_nim import FixedRuleNim
from random_agent import RandomAgent
from minmax import MinMaxAgent
from lib import Nim

logging.basicConfig(level=logging.INFO)

def hash_list(l):
    '''
    Hashes a list of integers into a string
    '''
    return "-".join([str(i) for i in l])


def unhash_list(l):
    '''
    Unhashes a string of integers into a list
    '''
    return [int(i) for i in l.split("-")]


def decay(value, decay_rate):
    return value * decay_rate


class NimRLMonteCarloAgent:
    def __init__(self, num_rows: int, epsilon: float = 0.3, alpha: float = 0.5, gamma: float = 0.9):
        """Initialize agent."""
        self.num_rows = num_rows
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.G = dict()
        self.state_history = []

    def get_action(self, state: Nim):
        """Return action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.choice(self.get_possible_actions(state))
            if (hash_list(state.rows), action) not in self.G:
                self.G[(hash_list(state.rows), action)] = random.uniform(1.0, 0.01)
            return action
        else:
            max_G = -math.inf
            best_action = None
            for r, c in enumerate(state.rows):
                for o in range(1, c+1):
                    if (hash_list(state.rows), (r, o)) not in self.G:
                        self.G[(hash_list(state.rows), (r, o))] = random.uniform(1.0, 0.01)
                        G = self.G[(hash_list(state.rows), (r, o))]
                    else:
                        G = self.G[(hash_list(state.rows), (r, o))]
                    if G > max_G:
                        max_G = G
                        best_action = (r, o)
            return best_action

    def update_state(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0

        for state, reward in reversed(self.state_history):
            self.G[state] = self.G.get(state, 0) + self.alpha * (target - self.G.get(state, 0))
            target += reward

        self.state_history = []
        self.epsilon -= 10e-5

    def compute_reward(self, state: Nim):
        return 0 if state.goal() else -1

    def get_possible_actions(self, state: Nim):
        actions = []
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                actions.append((r, o))
        return actions

    def get_G(self, state: Nim, action: tuple):
        return self.G.get((hash_list(state.rows), action), 0)

    def battle(self, opponent, training=True):
        player = 0
        agent_wins = 0
        for episode in range(rounds):
            self.current_state = Nim(num_rows=self.num_rows)
            while True:
                if player == 0:
                    action = self.get_action(self.current_state)
                    self.current_state.nimming_remove(*action)
                    reward = self.compute_reward(self.current_state)
                    self.update_state(hash_list(self.current_state.rows), reward)
                    player = 1
                else:
                    action = opponent(self.current_state)
                    self.current_state.nimming_remove(*action)
                    player = 0

                if self.current_state.goal():
                    logging.info("Player {} wins!".format(1 - player))
                    break

            winner = 1 - player
            if winner == 0:
                agent_wins += 1
            # episodic learning
            self.learn()

            if episode % 1000 == 0:
                logging.info("Win rate: {}".format(agent_wins / (episode + 1)))
        if not training:
            logging.info("Win rate: {}".format(agent_wins / rounds))

class NimRLTemporalDifferenceAgent:
    """
    An agent that learns to play Nim through temporal difference learning.
    """
    def __init__(self, num_rows: int, epsilon: float = 0.4, alpha: float = 0.3, gamma: float = 0.9):
        """Initialize agent."""
        self.num_rows = num_rows
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.Q = dict()

    def init_reward(self, state: Nim):
        '''Initialize reward for every state and every action with a random value'''
        for i in range(1, state.num_rows):
            nim = Nim(num_rows=i)
            for r, c in enumerate(nim.rows):
                for o in range(1, c+1):
                    self.set_Q(hash_list(nim.rows), (r, o),
                               np.random.uniform(0, 0.01))

    def get_Q(self, state: Nim, action: tuple):
        """Return Q-value for state and action."""
        if (hash_list(state.rows), action) in self.Q:
            logging.info("Getting Q for state: {} and action: {}".format(hash_list(state.rows), action))
            logging.info("Q-value: {}".format(self.Q[(hash_list(state.rows), action)]))
            return self.Q[(hash_list(state.rows), action)]
        else:
            # initialize Q-value for state and action
            self.set_Q(hash_list(state.rows), action, np.random.uniform(0, 0.01))
            return self.Q[(hash_list(state.rows), action)]

    def set_Q(self, state: str, action: tuple, value: float):
        """Set Q-value for state and action."""
        # logging.info("Setting Q for state: {} and action: {} to value: {}".format(state, action, value))
        self.Q[(state, action)] = value

    def get_max_Q(self, state: Nim):
        """Return maximum Q-value for state."""
        max_Q = -math.inf
        # logging.info(state.rows)
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                # logging.info("Just Q: {}".format(self.get_Q(state, (r, o))))
                max_Q = max(max_Q, self.get_Q(state, (r, o)))
        # logging.info("Max Q: {}".format(max_Q))
        return max_Q

    def get_average_Q(self, state: Nim):
        """Return average Q-value for state."""
        total_Q = 0
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                total_Q += self.get_Q(state, (r, o))
        return total_Q / len(state.rows)

    def get_possible_actions(self, state: Nim):
        """Return all possible actions for state."""
        possible_actions = []
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                possible_actions.append((r, o))
        return possible_actions

    def get_action(self, state: Nim):
        """Return action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            logging.info("Getting best action")
            max_Q = -math.inf
            best_action = None
            for r, c in enumerate(state.rows):
                for o in range(1, c+1):
                    Q = self.get_Q(state, (r, o))
                    if Q > max_Q:
                        max_Q = Q
                        best_action = (r, o)
            return best_action

    def register_state(self, state: Nim):
        # for each possible move in state, initialize random Q value
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                if (hash_list(state.rows), (r, o)) not in self.Q:
                    val = np.random.uniform(0, 0.01)
                    # logging.info("Registering state: {} and action: {} to {}".format(state.rows, (r, o), val))
                    self.set_Q(hash_list(state.rows), (r, o), val)
                else:
                    logging.info("State already registered: {} and action: {}".format(state.rows, (r, o)))

    def update_Q(self, reward: int, game_over: bool):
        """Update Q-value for previous state and action."""

        if game_over:
            # self.set_Q(hash_list(self.previous_state.rows), self.previous_action, reward)
            self.set_Q(hash_list(self.previous_state.rows), self.previous_action, self.get_Q(self.previous_state, self.previous_action) + self.alpha * (reward - self.get_Q(self.previous_state, self.previous_action)))

        else:
        # if reward != -1:
            self.register_state(self.current_state)
            if self.previous_action is not None:
                self.set_Q(hash_list(self.previous_state.rows), self.previous_action, self.get_Q(self.previous_state, self.previous_action) +
                           self.alpha * (reward + self.gamma) * (self.get_max_Q(self.current_state) - self.get_Q(self.previous_state, self.previous_action)))
        # else:
        #     self.set_Q(hash_list(self.previous_state.rows), self.previous_action, self.get_Q(self.previous_state, self.previous_action) + self.alpha * (reward - self.get_Q(self.previous_state, self.previous_action)))

    def print_best_action_for_each_state(self):
        for state in self.Q:
            logging.info("State: {}".format(state[0]))
            nim = Nim(5)
            nim.rows = unhash_list(state[0])
            logging.info("Best action: {}".format(self.choose_action(nim)))

    def test_against_random(self, round, random_agent):
        wins = 0
        for i in range(rounds):
            nim = Nim(num_rows=5)
            player = 0
            while not nim.goal():
                if player == 0:
                    move = self.choose_action(nim)
                    # logging.info(f"Reinforcement move: Removed {move[1]} objects from row {move[0]}")
                    nim.nimming_remove(*move)
                else:
                    move = random_agent(nim)
                    # logging.info(f"Random move {random_agent.num_moves}: Removed {move[1]} objects from row {move[0]}")
                    nim.nimming_remove(*move)
                player = 1 - player

            winner = 1 - player
            if winner == 0:
                wins += 1

        logging.info(f"Win Rate in round {round}: {wins / rounds}")

    def battle(self, agent, rounds=1000, training=True, momentary_testing=False):
        """Train agent by playing against other agents."""
        agent_wins = 0
        winners = []
        for episode in range(rounds):
            # logging.info(f"Episode {episode}")
            nim = Nim(num_rows=5)
            self.current_state = nim
            self.previous_state = None
            self.previous_action = None
            player = 0
            while True:
                reward = 0
                if player == 0:
                    self.previous_state = deepcopy(self.current_state)
                    self.previous_action = self.get_action(self.current_state)
                    self.current_state.nimming_remove(
                        *self.previous_action)
                    player = 1
                else:
                    move = agent(self.current_state)
                    # logging.info("Random agent move: {}".format(move))
                    self.current_state.nimming_remove(*move)
                    player = 0

                # learning by calculating reward for the current state
                if self.current_state.goal():
                    winner = 1 - player
                    if winner == 0:
                        logging.info("Agent won")
                        agent_wins += 1
                        reward = 1
                    else:
                        logging.info("Random won")
                        reward = -1
                    winners.append(winner)
                    self.update_Q(reward, self.current_state.goal())
                    break
                else:
                    self.update_Q(reward, self.current_state.goal())

            # decay epsilon after each episode
            self.epsilon = self.epsilon - 0.1 if self.epsilon > 0.1 else 0.1
            self.alpha *= -0.0005
            if self.alpha < 0.1:
                self.alpha = 0.1

            if training and momentary_testing:
                if episode % 100 == 0:
                    logging.info(f"Episode {episode} finished, sampling")
                    random_agent = BrilliantEvolvedAgent()
                    self.test_against_random(
                        episode, random_agent.random_agent)

        if not training:
            logging.info("Reinforcement agent won {} out of {} games".format(
                agent_wins, rounds))
        # self.print_best_action_for_each_state()
        return winners

    def choose_action(self, state: Nim):
        """Return action based on greedy policy."""
        max_Q = -math.inf
        best_action = None
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                Q = self.get_Q(state, (r, o))
                if Q > max_Q:
                    max_Q = Q
                    best_action = (r, o)
        if best_action is None:
            return random.choice(self.get_possible_actions(state))
        else:
            return best_action

if __name__ == "__main__":
    rounds = 10000
    minmax_wins = 0

    nim = Nim(num_rows=5)
    agent_tda = NimRLTemporalDifferenceAgent(num_rows=5)
    random_agent = RandomAgent()

    # agentG = NimRLMonteCarloAgent(num_rows=7)
    agent_tda.battle(random_agent.play, rounds=10000)
    agent_tda.epsilon = 0.1

    # TESTING
    logging.info("Testing against random agent")
    agent_tda.battle(random_agent.random_agent, training=False, rounds=1000)
