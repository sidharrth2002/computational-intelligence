'''
In this file, I will try to implement Nim where there is an evolved set of rules/strategies.
For each scenario, I will have a set of rules that will be used to determine the best move.
The rules currently are:
1. If one pile, take $x$ number of sticks from the pile.
2. If two piles:
    a. If 1 pile has 1 stick, take x sticks
    b. If 2 piles have multiple sticks, take x sticks from the larger pile
3. If three piles and two piles have the same size, remove all sticks from the smallest pile
4. If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles
'''

from collections import namedtuple
from copy import deepcopy
from itertools import accumulate
from operator import xor
import random

class Nim:
    def __init__(self, num_rows: int, k: int = None):
        self.num_rows = num_rows
        self._k = k
        self.rows = [i*2+1 for i in range(num_rows)]

    def nimming_remove(self, row: int, num_objects: int):
        assert self.rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self.rows[row] -= num_objects

    def goal(self):
        return sum(self.rows) == 0

Nimply = namedtuple("Nimply", "row, num_objects")

class BrilliantEvolvedAgent:
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

    def init_population(self, population_size, nim: Nim):
        '''
        Initialize population of genomes,
        key is rule, value is number of sticks to take
        The rules currently are:
        1. If one pile, take $x$ number of sticks from the pile.
        2. If two piles:
            a. If 1 pile has 1 stick, take x sticks
            b. If 2 piles have multiple sticks, take x sticks from the larger pile
        3. If three piles and two piles have the same size, remove all sticks from the smallest pile
        4. If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles
        '''
        population = []
        for i in range(population_size):
            individual = {
                'rule_1': random.randint(1, nim.rows[0]),
                'rule_2a': random.randint(1, min(nim.rows)),
                'rule_2b': random.randint(1, max(nim.rows)),
                'rule_3': min(nim.rows),
                'rule_4': max(nim.rows) - min(nim.rows)
            }
            population.append(individual)
        return population

    def statistics(self, nim: Nim):
        '''
        Similar to Squillero's cooked function to get possible moves
        and statistics on Nim board
        '''
        stats = {
            'possible_moves': [(row, num_objects) for row in range(nim.num_rows) for num_objects in range(1, nim.rows[row]+1)],
            'active_rows_number': sum(o > 0 for o in nim.rows),
            'shortest_row': min((x for x in enumerate(nim.rows) if x[1] > 0), key=lambda y: y[1])[1],
            'longest_row': max((x for x in enumerate(nim.rows)), key=lambda y: y[1])[1],
            # only 1-stick row and not all rows having only 1 stick
            'row_with_1_stick_bool': (1 in nim.rows) and not all(x == 1 for x in nim.rows),
            'nim_sum': self.nim_sum(nim)
        }

        brute_force = []
        for move in stats['possible_moves']:
            tmp = deepcopy(nim)
            tmp.nimming_remove(*move)
            brute_force.append((move, self.nim_sum(tmp)))
        stats['brute_force'] = brute_force

        return stats

    def strategy(self, nim: Nim):
        '''
        Returns the best move to make based on the statistics
        '''
        stats = self.statistics(nim)
        if stats['active_rows_number'] == 1:
            # take all sticks from the only row
            return (0, stats['shortest_row'])
        elif stats['active_rows_number'] == 2:
            if stats['row_with_1_stick_bool']:
                # take all sticks from the row with 1 stick
                return Nimply(stats['possible_moves'][0][0], stats['possible_moves'][0][1])
            else:
                # take all sticks from the largest row
                return Nimply(stats['longest_row'][0], stats['longest_row'][1])
        elif stats['active_rows_number'] == 3:
            if stats['shortest_row'] == stats['nim_sum']:
                # take all sticks from the smallest row
                return Nimply(stats['shortest_row'][0], stats['shortest_row'][1])
            else:
                # take all sticks from the largest row
                return Nimply(stats['longest_row'][0], stats['longest_row'][1])
        else:
            # take all sticks from the smallest row until it is the same size as the other rows
            return Nimply(stats['shortest_row'][0], stats['shortest_row'][1])

    def random_agent(self, nim: Nim):
        '''
        Random agent that takes a random move
        '''
        stats = self.statistics(nim)
        return random.choice(stats['possible_moves'])

    def calculate_fitness(self, genome):
        '''
        Calculate fitness by playing the genome's strategy against a random agent
        (cannot use nim sum agent as it is too good)
        '''

