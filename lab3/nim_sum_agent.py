from copy import deepcopy
from itertools import accumulate
from operator import xor
import random

from lib import Nim

# 3.1: Agent Using Fixed Rules
class ExpertFixedRuleAgent:
    '''
    Play the game of Nim using a fixed rule 
    (always leave nim-sum at the end of turn)
    '''
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
        # return sum([i^r for i, r in enumerate(nim._rows)])
    
    def play(self, nim: Nim):
        # remove objects from row to make nim-sum 0
        nim_sum = self.nim_sum(nim)
        all_possible_moves = [(r, o) for r, c in enumerate(nim.rows) for o in range(1, c+1)]     
        move_found = False
        for move in all_possible_moves:
            replicated_nim = deepcopy(nim)
            replicated_nim.nimming_remove(*move)
            if self.nim_sum(replicated_nim) == 0:
                nim.nimming_remove(*move)
                move_found = True
                break
        # if a valid move not found, return random move
        if not move_found:
            move = random.choice(all_possible_moves)
            nim.nimming_remove(*move)
        
        # print(f"Move {self.num_moves}: Removed {move[1]} objects from row {move[0]}")
        self.num_moves += 1