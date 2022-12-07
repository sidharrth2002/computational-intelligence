from copy import deepcopy
from functools import lru_cache
from itertools import accumulate
import math
from operator import xor
from evolved_nim import BrilliantEvolvedAgent

from lib import Nim

class MinMaxAgent:
    def __init__(self):
        self.num_moves = 0
    
    def nim_sum(self, nim: Nim):
        '''
        Returns the nim sum of the current game board
        by taking an XOR of all the rows.
        Ideally, agent should try to leave nim sum of 0 at the end of turn
        '''
        *_, result = accumulate(nim.rows, xor)
        return result

    def evaluate(self, nim: Nim, is_maximizing: bool):
        '''
        Returns the evaluation of the current game board
        '''
        if all(row == 0 for row in nim.rows):
            return -1 if is_maximizing else 1

    def minmax(self, nim: Nim, depth: int, maximizing_player: bool, alpha: int = -1, beta: int = 1, max_depth: int = 3):
        '''
        Depth-limited Minimax algorithm to find the best move with alpha-beta pruning and depth limit
        '''
        if depth == 0 or nim.goal():
            return self.evaluate(nim, maximizing_player)

        if maximizing_player:
            value = -math.inf
            for r, c in enumerate(nim.rows):
                for o in range(1, c+1):
                    # make copy of nim object before running a nimming operation
                    replicated_nim = deepcopy(nim)
                    replicated_nim.nimming_remove(r, o)
                    value = max(value, self.minmax(replicated_nim, depth-1, False, alpha, beta))
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
            return value
        else:
            value = math.inf
            for r, c in enumerate(nim.rows):
                for o in range(1, c+1):
                    # make copy of nim object before running a nimming operation
                    replicated_nim = deepcopy(nim)
                    replicated_nim.nimming_remove(r, o)
                    value = min(value, self.minmax(replicated_nim, depth-1, True, alpha, beta))
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
            return value

    def play(self, nim: Nim):
        '''
        Agent returns the best move based on minimax algorithm
        '''
        possible_moves = []
        for r, c in enumerate(nim.rows):
            for o in range(1, c+1):
                # make copy of nim object before running a nimming operation
                replicated_nim = deepcopy(nim)
                replicated_nim.nimming_remove(r, o)
                possible_moves.append((r, o, self.minmax(replicated_nim, 500, False)))
        # sort possible moves by the value returned by minimax
        possible_moves.sort(key=lambda x: x[2], reverse=True)
        # return the best move
        return possible_moves[0][0], possible_moves[0][1]

        # best_move = None
        # best_value = -math.inf
        # for r, c in enumerate(nim.rows):
        #     for o in range(1, c+1):
        #         replicated_nim = deepcopy(nim)
        #         replicated_nim.nimming_remove(r, o)
        #         value = self.minmax(replicated_nim, 30, True)
        #         if value > best_value:
        #             best_value = value
        #             best_move = (r, o)
        # self.num_moves += 1
        # return best_move

rounds = 10

minmax_wins = 0
for i in range(rounds):
    nim = Nim(num_rows=3)
    agent = MinMaxAgent()
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