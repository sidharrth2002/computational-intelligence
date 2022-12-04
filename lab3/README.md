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

I have discussed with Karl Wennerstrom and Diego Gasco.