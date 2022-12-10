from collections import Counter
from copy import deepcopy
from itertools import accumulate
from operator import xor
import random
from typing import Callable

from lib import Genome, Nim, Nimply


class FixedRuleNim:
    def __init__(self):
        self.num_moves = 0
        self.OFFSPRING_SIZE = 30
        self.POPULATION_SIZE = 100
        self.GENERATIONS = 100
        self.nim_size = 5

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
            a. If 1 pile has 1 stick, wipe out the pile
            b. If 2 piles have multiple sticks, take x sticks from any pile
        3. If three piles and two piles have the same size, remove all sticks from the smallest pile
        4. If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles
        '''
        population = []
        for i in range(population_size):
            # rules 3 and 4 are fixed (apply for 3 or more piles)
            # different strategies for different rules (situations on the board)
            individual = {
                'rule_1': [0, random.randint(0, (nim.num_rows - 1) * 2)],
                'rule_2a': [random.randint(0, 1), random.randint(0, (nim.num_rows - 1) * 2)],
                'rule_2b': [random.randint(0, 1), random.randint(0, (nim.num_rows - 1) * 2)],
                'rule_3': [nim.rows.index(min(nim.rows)), min(nim.rows)],
                'rule_4': [nim.rows.index(max(nim.rows)), max(nim.rows) - min(nim.rows)]
            }
            genome = Genome(individual)
            population.append(genome)
        return population

    def statistics(self, nim: Nim):
        '''
        Similar to Squillero's cooked function to get possible moves
        and statistics on Nim board
        '''
        # print('In statistics')
        # print(nim.rows)
        stats = {
            'possible_moves': [(r, o) for r, c in enumerate(nim.rows) for o in range(1, c + 1) if nim.k is None or o <= nim.k],
            # 'possible_moves': [(row, num_objects) for row in range(nim.num_rows) for num_objects in range(1, nim.rows[row]+1)],
            'num_active_rows': sum(o > 0 for o in nim.rows),
            'shortest_row': min((x for x in enumerate(nim.rows) if x[1] > 0), key=lambda y: y[1])[0],
            'longest_row': max((x for x in enumerate(nim.rows)), key=lambda y: y[1])[0],
            # only 1-stick row and not all rows having only 1 stick
            '1_stick_row': any([1 for x in nim.rows if x == 1]) and not all([1 for x in nim.rows if x == 1]),
            'nim_sum': self.nim_sum(nim)
        }

        brute_force = []
        for move in stats['possible_moves']:
            tmp = deepcopy(nim)
            tmp.nimming_remove(*move)
            brute_force.append((move, self.nim_sum(tmp)))
        stats['brute_force'] = brute_force

        return stats

    def strategy(self):
        '''
        Returns the best move to make based on the statistics
        '''
        def engine(nim: Nim):
            stats = self.statistics(nim)
            if stats['num_active_rows'] == 1:
                # print('m1')
                return Nimply(stats['shortest_row'], random.randint(1, stats['possible_moves'][0][1]))
            elif stats["num_active_rows"] % 2 == 0:
                # print('m2')
                if max(nim.rows) == 1:
                    return Nimply(stats['longest_row'], 1)
                else:
                    pile = random.choice([i for i, x in enumerate(nim.rows) if x > 1])
                    return Nimply(pile, nim.rows[pile] - 1)
            elif stats['num_active_rows'] == 3:
                # print('m3')
                unique_elements = set(nim.rows)
                # check if 2 rows have the same number of sticks
                two_rows_with_same_elements = False
                for element in unique_elements:
                    if nim.rows.count(element) == 2:
                        two_rows_with_same_elements = True
                        break

                if len(nim.rows) == 3 and two_rows_with_same_elements:
                    # remove 1 stick from the longest row
                    print(nim.rows)
                    return Nimply(stats['longest_row'], max(max(nim.rows) - nim.rows[stats['shortest_row']], 1))
                else:
                    # do something random
                    return Nimply(*random.choice(stats['possible_moves']))
            elif stats['num_active_rows'] >= 4:
                # print('m4')
                counter = Counter()
                for element in nim.rows:
                    counter[element] += 1
                if len(counter) == 2:
                    if counter.most_common()[0][1] == 1:
                        # remove x sticks from the smallest pile until it is the same size as the other piles
                        return Nimply(stats['shortest_row'], max(nim.rows[stats['shortest_row']] - counter.most_common()[1][0], 1))
                return random.choice(stats['possible_moves'])
            else:
                # print('m5')
                return random.choice(stats['possible_moves'])
        return engine

    def random_agent(self, nim: Nim):
        '''
        Random agent that takes a random move
        '''
        stats = self.statistics(nim)
        return random.choice(stats['possible_moves'])

if __name__ == '__main__':
    rounds = 20
    evolved_agent_wins = 0
    for i in range(rounds):
        nim = Nim(5)
        orig = nim.rows
        fixedrule = FixedRuleNim()
        engine = fixedrule.strategy()

        # play against random
        player = 0
        while not nim.goal():
            if player == 0:
                move = engine(nim)
                print('move of player 1: ', move)
                nim.nimming_remove(*move)
                player = 1
                print("After Player 1 made move: ", nim.rows)
            else:
                move = fixedrule.random_agent(nim)
                print('move of player 2: ', move)
                nim.nimming_remove(*move)
                player = 0
                print("After Player 2 made move: ", nim.rows)
        winner = 1 - player
        if winner == 0:
            evolved_agent_wins += 1
    print(f'Fixed rule agent won {evolved_agent_wins} out of {rounds} games')