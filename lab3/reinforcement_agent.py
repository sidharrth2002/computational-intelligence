
from copy import deepcopy
import math
import os
import random

from matplotlib import pyplot as plt
from evolved_nim import BrilliantEvolvedAgent
from lib import Nim

def hash_list(l):
    return "-".join([str(i) for i in l])

def unhash_list(l):
    return [int(i) for i in l.split("-")]

class NimReinforcementAgent:
    """An agent that learns to play Nim through reinforcement learning."""

    def __init__(self, num_rows: int, epsilon: float = 0.8, alpha: float = 0.5, gamma: float = 0.8):
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
        '''Initialize reward for state with random uniform values'''
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                self.set_Q(hash_list(state.rows), (r, o), random.uniform(1.0, 0.1))

    def get_Q(self, state: Nim, action: tuple):
        """Return Q-value for state and action."""
        if (state, action) in self.Q:
            return self.Q[(hash_list(state.rows), action)]
        else:
            return 0

    def set_Q(self, state: str, action: tuple, value: float):
        """Set Q-value for state and action."""
        print("Setting Q for state: {} and action: {} to value: {}".format(state, action, value))
        self.Q[(state, action)] = value

    def get_max_Q(self, state: Nim):
        """Return maximum Q-value for state."""
        max_Q = -math.inf
        # print(state.rows)
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                # print("Just Q: {}".format(self.get_Q(state, (r, o))))
                max_Q = max(max_Q, self.get_Q(state, (r, o)))
        # print("Max Q: {}".format(max_Q))
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
            max_Q = -math.inf
            best_action = None
            for r, c in enumerate(state.rows):
                for o in range(1, c+1):
                    Q = self.get_Q(state, (r, o))
                    if Q > max_Q:
                        max_Q = Q
                        best_action = (r, o)
            return best_action


    def update_Q(self, reward: int, game_over: bool):
        """Update Q-value for previous state and action."""
        # print("Previous state: {}".format(self.previous_state.rows))
        # print("Previous action: {}".format(self.previous_action))
        # new_value = hash_list(self.previous_state.rows), self.previous_action, self.get_Q(self.previous_state, self.previous_action) + self.alpha * (reward + self.gamma * self.get_max_Q(self.current_state) - self.get_Q(self.previous_state, self.previous_action))
        # Q(s, a) <- old value estimate + alpha * (new value estimate - old value estimate)

        # If the game was over, new_state is not a valid state.
        if game_over:
            self.set_Q(hash_list(self.previous_state.rows), self.previous_action, reward)
        else:
            # If the game is not over, then it sets the expected value of the action in the previous state to be the reward, plus the average value of the actions in the new state.
            self.set_Q(hash_list(self.previous_state.rows), self.previous_action, reward + self.gamma * self.get_average_Q(self.current_state))

            # self.q[state][action] = self.q[state][action] + learning_rate * (reward + discount_factor * max(self.q[new_state]) - self.q[state][action])
            # self.set_Q(hash_list(self.previous_state.rows), self.previous_action, self.get_Q(self.previous_state, self.previous_action) + self.alpha * (reward + self.gamma * self.get_max_Q(self.current_state) - self.get_Q(self.previous_state, self.previous_action)))

            # old_value_estimate = self.get_Q(self.previous_state, self.previous_action)
            # new_value_estimate = reward + self.gamma * self.get_max_Q(self.current_state)
            # new_q = old_value_estimate + self.alpha * (new_value_estimate - old_value_estimate)
            # new_value = hash_list(self.previous_state.rows), self.previous_action, new_q
            # self.set_Q(*new_value)

        # old_value_estimate = self.get_Q(self.previous_state, self.previous_action)
        # new_value_estimate = reward + self.gamma * self.get_max_Q(self.current_state)
        # new_q = old_value_estimate + self.alpha * (new_value_estimate - old_value_estimate)
        # new_value = hash_list(self.previous_state.rows), self.previous_action, new_q
        # # print("New value: {}".format(new_value))
        # self.set_Q(*new_value)

    def calculate_reward(self, state: Nim):
        """Return reward for state."""
        if state.goal():
            return 1
        else:
            return -1

    def learn_play_random(self, random_agent):
        """Train agent by playing against a random agent."""
        agent_wins = 0
        winners = []
        for episode in range(500):
            nim = Nim(num_rows=3)
            self.current_state = nim
            self.previous_state = None
            self.previous_action = None
            player = 0
            while True:
                reward = 0
                if player == 0:
                    self.previous_state = deepcopy(self.current_state)
                    # print("Current state: {}".format(self.current_state.rows))
                    self.previous_action = self.get_action(self.current_state)
                    # if self.previous_action is None:
                    #     # make random move if no possible actions
                    #     # print("Making random move")
                    #     self.current_state.nimming_remove(*random.choice(self.get_possible_actions(self.current_state)))
                    # else:
                        # print("RL action: {}".format(self.previous_action))
                    self.current_state.nimming_remove(*self.previous_action)
                    player = 1
                else:
                    move = random_agent.random_agent(self.current_state)
                    # print("Random agent move: {}".format(move))
                    self.current_state.nimming_remove(*move)
                    player = 0

                # learning by calculating reward for the current state

                if self.current_state.goal():
                    winner = 1 - player
                    if winner == 0:
                        print("Agent won")
                        agent_wins += 1
                        reward = 1
                    else:
                        print("Random won")
                        reward = -1
                    winners.append(winner)
                    self.update_Q(reward, self.current_state.goal())
                    break
                else:
                    self.update_Q(reward, self.current_state.goal())

        # decay epsilon after each episode
        self.epsilon = self.epsilon - 10e-5 if self.epsilon > 0.1 else 0.1

        print("Agent won {} out of {} games".format(agent_wins, 5000))
        print(self.Q)
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
            print("Making random move")
            print(self.Q)
            # make random move if no possible actions
            return random.choice(self.get_possible_actions(state))
        else:
            # print(f"Best action in {state.rows}: Removed {best_action[1]} objects from row {best_action[0]}")
            return best_action

rounds = 20
minmax_wins = 0

nim = Nim(num_rows=4)
print(nim.rows)
agent = NimReinforcementAgent(num_rows=4)
random_agent = BrilliantEvolvedAgent()
agent.init_reward(nim)
winners = agent.learn_play_random(random_agent)

# plot winners for each episode and save to file

plt.plot(winners)
plt.savefig("winners.png")

# for i in range(rounds):
#     nim = Nim(num_rows=5)
#     player = 1
#     while not nim.goal():
#         if player == 0:
#             move = agent.choose_action(nim)
#             print(f"Reinforcement move: Removed {move[1]} objects from row {move[0]}")
#             nim.nimming_remove(*move)
#             print(nim.rows)
#         else:
#             move = random_agent.random_agent(nim)
#             print(f"Random move {random_agent.num_moves}: Removed {move[1]} objects from row {move[0]}")
#             nim.nimming_remove(*move)
#             print(nim.rows)
#         player = 1 - player

#     winner = 1 - player
#     if winner == 0:
#         minmax_wins += 1
#     # player that made the last move wins
#     print(f"Player {winner} wins in round {i+1}!")

# print(f"Reinforcement agent wins {minmax_wins} out of {rounds} rounds")