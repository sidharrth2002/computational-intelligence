from copy import deepcopy
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

    def minmax(self, nim: Nim, depth: int, maximizing_player: bool):
        '''
        Minimax algorithm to find the best move
        '''
        if depth == 0 or nim.goal():
            return self.nim_sum(nim)
        
        if maximizing_player:
            value = -math.inf
            for r, c in enumerate(nim.rows):
                for o in range(1, c+1):
                    # make copy of nim object before running a nimming operation
                    replicated_nim = deepcopy(nim)
                    replicated_nim.nimming_remove(r, o)
                    value = max(value, self.minmax(replicated_nim, depth-1, False))
            return value
        else:
            value = math.inf
            for r, c in enumerate(nim.rows):
                for o in range(1, c+1):
                    # make copy of nim object before running a nimming operation
                    replicated_nim = deepcopy(nim)
                    replicated_nim.nimming_remove(r, o)
                    value = min(value, self.minmax(replicated_nim, depth-1, True))
            return value

    def play(self, nim: Nim):
        '''
        Minimax algorithm to find the best move
        '''
        best_move = None
        best_value = -math.inf
        for r, c in enumerate(nim.rows):
            for o in range(1, c+1):
                replicated_nim = deepcopy(nim)
                replicated_nim.nimming_remove(r, o)
                value = self.minmax(replicated_nim, 5, False)
                if value > best_value:
                    best_value = value
                    best_move = (r, o)
        self.num_moves += 1
        return best_move

nim = Nim(num_rows=5)
agent = MinMaxAgent()
random_agent = BrilliantEvolvedAgent()
player = 0
while not nim.goal():
    if player == 0:
        move = agent.play(nim)
        nim.nimming_remove(*move)
    else:
        move = random_agent.random_agent(nim)
        nim.nimming_remove(*move)
    player = 1 - player
print(f"Player {player} wins!")