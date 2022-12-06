
class NimReinforcementAgent:
    """An agent that learns to play Nim through reinforcement learning."""
    def __init__(self, environment):
        """Initialize the agent."""
        self.environment = environment
        self.state = environment.get_state()
        self.actions = environment.actions
        self.q_table = np.zeros((len(self.actions), 1))
        self.alpha = 0.1
        self.gamma = 0.9
        
    def choose_action(self):
        """Choose an action based on the current state."""
        if np.random.uniform(0, 1) < 0.1:
            # Explore a random action with probability 0.1
            action = np.random.choice(self.actions)
        else:
            # Choose the action with the highest Q-value with probability 0.9
            action = self.actions[np.argmax(self.q_table)]
        return action

    def update_q_table(self, action, reward):
        """Update the Q-table based on the action and the reward."""
        self.q_table[self.actions.index(action)] = (1 - self.alpha) * self.q_table[self.actions.index(action)] + self.alpha * (reward + self.gamma * np.max(self.q_table))
        
    def train(self, iterations):
        """Train the agent."""
        for step in range(iterations):
            # Choose an action based on the current state
            action = self.choose_action()

            # Take the action and receive a reward
            reward = self.environment.take_action(action)

            # Update the Q-table based on the action and the reward
            self.update_q_table(action, reward)

    def play(self, iterations):
