# 3.2: Agent Using Evolved Rules (Randomly Chooses Between Strategies)
from itertools import accumulate
from operator import xor
import random
import numpy as np

from lib import Nim

class EvolvedAgent1:
    '''
    Plays Nim using a set of rules that are evolved
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

    def play_nim(self, nim: Nim, prob_list: list):
        '''
        GA can choose between the following strategies:
        1. Randomly pick any row and any number of elements from that row
        2. Pick the shortest row
        3. Pick the longest row
        4. Pick based on the nim-sum of the current game board
        '''
        all_possible_moves = [(r, o) for r, c in enumerate(nim.rows) for o in range(1, c+1)]
        strategies = {
            'nim_sum': random.choice([move for move in all_possible_moves if self.nim_sum(deepcopy(nim).nimming_remove(*move)) == 0]),
            'random': random.choice(all_possible_moves),
            'all_elements_shortest_row': (nim.rows.index(min(nim.rows)), min(nim.rows)),
            '1_element_shortest_row': (nim.rows.index(min(nim.rows)), 1),
            'random_element_shortest_row': (nim.rows.index(min(nim.rows)), random.randint(1, min(nim.rows))),
            'all_elements_longest_row': (nim.rows.index(max(nim.rows)), max(nim.rows)),
            '1_element_longest_row': (nim.rows.index(max(nim.rows)), 1),
            'random_element_longest_row': (nim.rows.index(max(nim.rows)), random.randint(1, max(nim.rows))),
        }
        
        p = random.random()
        strategy = None
        if p < prob_list[0]:
            strategy = strategies['random']
        elif p >= prob_list[0] and p < prob_list[1]:
            strategy = random.choice([strategies['all_elements_shortest_row'], strategies['1_element_shortest_row'], strategies['random_element_shortest_row']])
        elif p >= prob_list[1] and p < prob_list[2]:
            strategy = random.choice([strategies['all_elements_longest_row'], strategies['1_element_longest_row'], strategies['random_element_longest_row']])
        else:
            strategy = strategies['nim_sum']
        
        nim.nimming_remove(*strategy)
        self.num_moves += 1
        return sum(nim.rows)

    def play(self, nim: Nim):
        '''
        Play the game of Nim using the evolved rules
        '''
        prob_list = [0.25, 0.5, 0.75, 1]
        prob_list = self.evolve_probabilities(nim, prob_list, 20, 5)
        self.play_nim(nim, prob_list)

    def crossover(self, p1, p2):
        '''
        Crossover between two parents
        '''
        return np.random.choice(p1 + p2, size=4, replace=True)    

    def evolve_probabilities(self, nim: Nim, prob_list: list, num_generations: int, num_children: int):
        '''
        Evolve the probabilities of the strategies
        '''
        # create initial population
        population = [prob_list for _ in range(num_children)]
        # create initial fitness scores
        fitness_scores = [self.play(nim, p) for p in population]
        # create initial parents
        parents = [population[i] for i in np.argsort(fitness_scores)[:2]]
        # create new population
        new_population = []
        for _ in range(num_generations):
            # create children
            for _ in range(num_children):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                # child = []
                # for i in range(len(parents[0])):
                #     # crossover between parents
                    
                #     child.append(random.choice(parents)[i])
                new_population.append(child)
            # create fitness scores
            fitness_scores = [self.play_nim(nim, p) for p in new_population]
            # create new parents
            parents = [new_population[i] for i in np.argsort(fitness_scores)[:2]]
            # create new population
            new_population = []
        return parents[0]