
from copy import deepcopy
import math
import random
from lab3.lib import Nim


class NimReinforcementAgent:
    """An agent that learns to play Nim through reinforcement learning."""

    def __init__(self, num_rows: int, epsilon: float = 0.1, alpha: float = 0.5, gamma: float = 0.9):
        """Initialize agent."""
        self.num_rows = num_rows
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.current_state = None
        self.previous_state = None
        self.previous_action = None
        self.Q = dict()

    def get_Q(self, state: Nim, action: tuple):
        """Return Q-value for state and action."""
        if (state, action) in self.Q:
            return self.Q[(state, action)]
        else:
            return 0

    def set_Q(self, state: Nim, action: tuple, value: float):
        """Set Q-value for state and action."""
        self.Q[(state, action)] = value

    def get_max_Q(self, state: Nim):
        """Return maximum Q-value for state."""
        max_Q = -math.inf
        for r, c in enumerate(state.rows):
            for o in range(1, c+1):
                max_Q = max(max_Q, self.get_Q(state, (r, o)))
        return max_Q

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

    def update_Q(self, reward: int):
        """Update Q-value for previous state and action."""
        self.set_Q(self.previous_state, self.previous_action, self.get_Q(self.previous_state, self.previous_action) + self.alpha * (reward + self.gamma * self.get_max_Q(self.current_state) - self.get_Q(self.previous_state, self.previous_action)))

    def learn(self, state: Nim, reward: int):
        """Learn from previous state, action, and reward."""
        if self.previous_state is not None:
            self.update_Q(reward)
        self.previous_state = deepcopy(state)
        self.previous_action = self.get_action(state)

    def play(self, state: Nim):
        """Return action based on learned Q-values."""
        return self.get_action(state)

rounds = 10
minmax_wins = 0
for i in range(rounds):
    nim = Nim(num_rows=3)
    agent = NimReinforcementAgent(num_rows=3)
    random_agent = BrilliantEvolvedAgent()
    player = 0
    while not nim.goal():
        if player == 0:
            move = agent.play(nim)
            print(f"Minmax move {agent.num_moves}: Removed {move[1]} objects from row {move[0]}")
            print(nim.rows)
            nim.nimming_remove(*move)
        else:
            move = random_agent.random_agent(nim)
            print(f"Random move {random_agent.num_moves}: Removed {move[1]} objects from row {move[0]}")
            print(nim.rows)
            nim.nimming_remove(*move)
        player = 1 - player

    winner = 1 - player
    if winner == 0:
        minmax_wins += 1
    # player that made the last move wins
    print(f"Player {winner} wins in round {i+1}!")

print(f"Minmax wins {minmax_wins} out of {rounds} rounds")