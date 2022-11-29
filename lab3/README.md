# Nim Policy Search

Nim is a simple game where two players take turns removing objects from a pile. The player who removes the last object loses. The game is described in detail [here](https://en.wikipedia.org/wiki/Nim). There is a mathematical strategy to win Nim, by ensuring you always leave the opponent with a nim-sum number of objects (groups of 1, 2 and 4).

In this notebook, we will play nim-sum using the following agents:
1. An agent using fixed rules based on nim-sum
2. An agent using evolved rules
3. An agent using minmax
4. An agent using reinforcement learning

> Sidharrth Nagappan, 2022