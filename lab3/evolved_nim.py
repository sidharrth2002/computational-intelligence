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

class Genome:
    def __init__(self, rules):
        self.rules = rules
        self.fitness = 0

class BrilliantEvolvedAgent:
    def __init__(self):
        self.num_moves = 0
        self.OFFSPRING_SIZE = 30
        self.POPULATION_SIZE = 100
        self.GENERATIONS = 100
        self.nim_size = 3

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
            b. If 2 piles have multiple sticks, take x sticks from the smallest pile
        3. If three piles and two piles have the same size, remove all sticks from the smallest pile
        4. If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles
        '''
        population = []
        for i in range(population_size):
            # rules 3 and 4 are fixed (apply for 3 or more piles)
            # different strategies for different rules (situations on the board)
            individual = {
                'rule_1': [0, random.randint(0, nim.rows[0])],
                'rule_2a': [random.randint(0, 1), random.randint(0, self.nim_size * 2)],
                'rule_2b': [random.randint(0, 1), random.randint(0, self.nim_size * 2)],
                'rule_3': [nim.rows.index(min(nim.rows)), min(nim.rows)],
                'rule_4': [nim.rows.index(max(nim.rows)), max(nim.rows) - min(nim.rows)]
            }
            genome = Genome(individual)
            population.append(genome)
        return population

    def crossover(self, parent1, parent2, crossover_rate):
        '''
        Crossover function to combine two parents into a child
        '''
        child = {}
        for rule in parent1.rules:
            if random.random() < crossover_rate:
                child[rule] = parent1.rules[rule]
            else:
                child[rule] = parent2.rules[rule]
        return Genome(child)

        # child = deepcopy(parent1)
        # for rule in child.rules:
        #     if random.random() < 0.5:
        #         child.rules[rule] = deepcopy(parent2.rules[rule])
        # return child

        # child = {}
        # for key in parent1.keys():
        #     child[key] = random.choice([parent1[key], parent2[key]])
        # return child

    def tournament_selection(self, population, tournament_size):
        '''
        Tournament selection to select the best genomes
        '''
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

    def mutate(self, genome: Genome, mutation_rate=0.5):
        '''
        Mutate the genome by changing one of the rules (can end up in something stupid like removing more sticks than there are, but this is checked in the strategy function)
        '''
        rule = random.choice(list(genome.rules.keys()))
        if random.random() < mutation_rate:
            genome.rules[rule] = [random.randint(0, 1), random.randint(0, self.nim_size * 2)]
        return genome
        # rule = random.choice(list(genome.keys()))
        # genome[rule] = random.randint(1, 10)

    def statistics(self, nim: Nim):
        '''
        Similar to Squillero's cooked function to get possible moves
        and statistics on Nim board
        '''
        # print('In statistics')
        # print(nim.rows)
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

    def strategy(self, genome: dict):
        '''
        Returns the best move to make based on the statistics
        '''
        def evolution(nim: Nim):
            stats = self.statistics(nim)
            if stats['active_rows_number'] == 1:
                # print("Entering rule 1")
                num_to_leave = genome.rules['rule_1'][1]
                # see which move will leave the most sticks
                most_destructive_move = max(stats['possible_moves'], key=lambda x: x[1])
                if num_to_leave >= most_destructive_move[1]:
                    # remove only 1 stick
                    return Nimply(most_destructive_move[0], 1)
                else:
                    # make the move that leaves the desired number of sticks
                    move = [(row, num_objects) for row, num_objects in stats['possible_moves'] if nim.rows[row] - num_objects == num_to_leave]
                    if len(move) > 0:
                        return Nimply(move[0][0], move[0][1])
                    else:
                        # make random move
                        return Nimply(*random.choice(stats['possible_moves']))

                # if (max(stats['possible_moves'], key=lambda x: x[1])[1] <= genome['rule_1']):
                #     return Nimply(stats['active_rows_index'], genome['rule_1'])
                # genome specifies number of sticks to leave behind
                # chosen_move = random.choice([(move, num_objects) for move, num_objects in stats['possible_moves'] if nim.rows[move] - num_objects == genome['rule_1']])
                # return Nimply(chosen_move[0], chosen_move[1])
            elif stats['active_rows_number'] == 2:
                print("Entering rule 2")
                # rule 2a
                if stats['row_with_1_stick_bool']:
                    # first row
                    if genome.rules['rule_2a'][0] == 0:
                        pile = random.choice([i for i, x in enumerate(nim.rows) if x == 1])
                        return Nimply(pile, 1)
                    # second row
                    elif genome.rules['rule_2a'][0] == 1:
                        # print(nim.rows)
                        pile = random.choice([i for i, x in enumerate(nim.rows) if x >= 1])
                        num_objects_to_remove = nim.rows[pile] - genome.rules['rule_2a'][1]
                        if num_objects_to_remove < 1:
                            # take at least 1 stick
                            num_objects_to_remove = 1
                        return Nimply(pile, num_objects_to_remove)
                # rule 2b
                else:
                    if genome.rules['rule_2b'][0] == 0:
                        row = nim.rows.index(stats['shortest_row'])
                        print('shortest row', row)
                    else:
                        row = nim.rows.index(stats['longest_row'])
                        print('longest row', row)
                    num_objects_to_remove = max(nim.rows[row] - genome.rules['rule_2b'][1], 1)
                    print('num_objects_to_remove', num_objects_to_remove)
                    return Nimply(row, num_objects_to_remove)
            elif stats['active_rows_number'] >= 3:
                # print("Entering rule 3")
                # fixed set of rules for 4 or more piles (nothing changes idk)

                unique_elements = set(nim.rows)
                # check if 2 rows have the same number of sticks
                two_rows_with_same_elements = False
                for element in unique_elements:
                    if nim.rows.count(element) == 2:
                        two_rows_with_same_elements = True
                        break

                if len(nim.rows) == 3 and two_rows_with_same_elements:
                    # remove 1 stick from the longest row
                    return Nimply(nim.rows.index(stats['longest_row']), max(nim.rows) - stats['shortest_row'])
                else:
                    # do something random
                    return Nimply(*random.choice(stats['possible_moves']))

                # if stats['shortest_row'] == stats['longest_row']:
                #     return Nimply(genome.rules['rule_3'][0], genome.rules['rule_3'][1])
                # else:
                #     return Nimply(genome.rules['rule_4'][0], genome.rules['rule_4'][1])

            # if stats['active_rows_number'] == 1:
            #     # take all sticks from the only row
            #     return (0, stats['shortest_row'])
            # elif stats['active_rows_number'] == 2:
            #     if stats['row_with_1_stick_bool']:
            #         # take all sticks from the row with 1 stick
            #         return Nimply(stats['possible_moves'][0][0], stats['possible_moves'][0][1])
            #     else:
            #         # take all sticks from the largest row
            #         return Nimply(stats['longest_row'][0], stats['longest_row'][1])
            # elif stats['active_rows_number'] == 3:
            #     if stats['shortest_row'] == stats['nim_sum']:
            #         # take all sticks from the smallest row
            #         return Nimply(stats['shortest_row'][0], stats['shortest_row'][1])
            #     else:
            #         # take all sticks from the largest row
            #         return Nimply(stats['longest_row'][0], stats['longest_row'][1])
            # else:
            #     # take all sticks from the smallest row until it is the same size as the other rows
            #     return Nimply(stats['shortest_row'][0], stats['shortest_row'][1])

        return evolution

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
        wins = 0
        for i in range(3):
            print("Start Game")
            nim = Nim(3)
            player = 0
            engine = self.strategy(genome)
            while not nim.goal():
                if player == 0:
                    move = engine(nim)
                    print('move of player 1: ', move)
                    nim.nimming_remove(*move)
                    player = 1
                    print("After Player 1 made move: ", nim.rows)
                else:
                    nim.nimming_remove(*self.random_agent(nim))
                    player = 0
                    print("After Player 2 made move: ", nim.rows)
            print("Game Over")
            winner = 1 - player
            if winner == 0:
                wins += 1
        return wins

    def select_survivors(self, population: list, num_survivors: int):
        '''
        Select the best genomes from the population
        '''
        return sorted(population, key=lambda x: x.fitness, reverse=True)[:num_survivors]

    def learn(self, population_size=100, mutation_rate=0.1, crossover_rate=0.7, nim: Nim = None):
        initial_population = self.init_population(population_size, nim)
        for genome in initial_population:
            genome.fitness = self.calculate_fitness(genome)
        for i in range(self.GENERATIONS):
            # print(f'Generation {i}')
            new_offspring = []
            for j in range(self.OFFSPRING_SIZE):
                parent1 = random.choice(initial_population)
                parent2 = random.choice(initial_population)
                child = self.crossover(parent1, parent2, crossover_rate)
                child = self.mutate(child)
                new_offspring.append(child)
            initial_population += new_offspring
            for genome in initial_population:
                print(genome.rules)
            initial_population = self.select_survivors(initial_population, population_size)
        best_strategy = initial_population[0]
        return best_strategy

rounds = 20
evolved_agent_wins = 0
for i in range(rounds):
    nim = Nim(3)
    orig = nim.rows
    brilliantagent = BrilliantEvolvedAgent()
    best_strategy = brilliantagent.learn(nim=nim)
    engine = brilliantagent.strategy(best_strategy)

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
            move = brilliantagent.random_agent(nim)
            print('move of player 2: ', move)
            nim.nimming_remove(*move)
            player = 0
            print("After Player 2 made move: ", nim.rows)
    winner = 1 - player
    if winner == 0:
        evolved_agent_wins += 1
print(f'Evolved agent won {evolved_agent_wins} out of {rounds} games')