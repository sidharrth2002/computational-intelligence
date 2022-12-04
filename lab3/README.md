# Nim Policy Search

Nim is a simple game where two players take turns removing objects from a pile. The player who removes the last object loses. The game is described in detail [here](https://en.wikipedia.org/wiki/Nim). There is a mathematical strategy to win Nim, by ensuring you always leave the opponent with a nim-sum number of objects (groups of 1, 2 and 4).

In this notebook, we will play nim-sum using the following agents:
1. An agent using fixed rules based on nim-sum
2. An agent using evolved rules
3. An agent using minmax
4. An agent using reinforcement learning

## Rules

We came up with multiple rules, through discussion with friends and through research papers that define fixed rules for playing Nim. There are currently 4 rules that we have implemented. The rules are as follows:
1. If one pile, take x number of sticks from the pile.
2. If two piles:
    a. If 1 pile has 1 stick, take x sticks
    b. If 2 piles have multiple sticks, take x sticks from the larger pile
3. If three piles and two piles have the same size, remove all sticks from the smallest pile
4. If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles

### Task 3.1: Fixed Rules

The above rules are applied directly. An if-else sequence decides which strategy to employ based on the current layout and statistics on the nim board.

### Task 3.2: Evolved Rules

#### Approach 1: Evolving strategies for different board layouts (Rule-Strategy Evolution)

The rules are evolved using a genetic algorithm. A dictionary of strategies is evolved. The key is the rule (scenario/antecedent). The value is the maximum number of sticks to leave on the board in this scenario.

Mutation essentially swaps the values in the dictionaries. Crossover takes two parents and randomly chooses strategies for different rules. **Intuitively, the machine tries to learn the best strategy for each scenario on the board.**

| Opponent 1 | Opponent 2 | Win Rate |
|------------|------------|----------|
| Evolved    | Random     | 70%      |

#### Approach 2: Evolving probabilities of choosing different strategies

Strategies were originally chosen based on probability thresholds and a random number. The list of probabilities (thresholds) are evolved using a genetic algorithm. **Intuitively, the machine tries to learn the best probability of choosing each strategy, regardless of the rule.**

Code for this in notebook (not refined yet).

### Task 3.3: Minmax

Done with alpha-beta pruning.

### Task 3.4: Reinforcement Learning

Pending

### Acknowledgements

I have discussed with Karl Wennerstrom and Diego Gasco.