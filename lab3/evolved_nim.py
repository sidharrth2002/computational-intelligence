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

from collections import Counter, namedtuple
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

    @property
    def k(self):
        return self._k

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
                'rule_1': [0, random.randint(0, (self.nim_size - 1) * 2)],
                'rule_2a': [random.randint(0, 1), random.randint(0, (self.nim_size - 1) * 2)],
                'rule_2b': [random.randint(0, 1), random.randint(0, (self.nim_size - 1) * 2)],
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

    def strategy(self, genome: dict):
        '''
        Returns the best move to make based on the statistics
        '''
        def evolution(nim: Nim):
            stats = self.statistics(nim)
            if stats['num_active_rows'] == 1:
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

            elif stats['num_active_rows'] == 2:
                print("Entering rule 2")
                # rule 2a
                if stats['1_stick_row']:
                    # if there is a 1-stick row, have to choose between wiping it out or taking from the other row
                    if genome.rules['rule_2a'][0] == 0:
                        # wipe out the 1-stick row
                        print('wiping out 1-stick row')
                        pile = [row for row in range(nim.num_rows) if nim.rows[row] == 1][0]
                        return Nimply(pile, 1)
                    else:
                        # take out the desired number of sticks from the other row
                        pile = random.choice([index for index, x in enumerate(nim.rows) if x > 1])
                        num_objects_to_remove = max(1, nim.rows[pile] - genome.rules['rule_2a'][1])
                        # move = [(row, num_objects) for row, num_objects in stats['possible_moves'] if nim.rows[row] - num_objects == genome.rules['rule_2a'][1]]
                        print(pile, num_objects_to_remove)
                        return Nimply(pile, num_objects_to_remove)
                # rule 2b
                # both piles have many elements, take from either the smallest or the largest pile
                else:
                    if genome.rules['rule_2b'][0] == 0:
                        # take from the smallest pile
                        pile = stats['shortest_row']
                        num_objects_to_remove = max(1, nim.rows[pile] - genome.rules['rule_2b'][1])
                        return Nimply(pile, num_objects_to_remove)
                    else:
                        # take from the largest pile
                        pile = stats['longest_row']
                        num_objects_to_remove = max(1, nim.rows[pile] - genome.rules['rule_2b'][1])
                        return Nimply(pile, num_objects_to_remove)
                    # return Nimply(genome.rules['rule_2b'][0], max(nim.rows[genome.rules['rule_2b'][0]] - genome.rules['rule_2b'][1], 1))
                    # if genome.rules['rule_2b'][1] == 0:
                    #     # take from the smallest pile
                    #     pile = stats['possible_moves'][0][0]
                    #     return Nimply(pile, nim.rows[pile])

                    # first row
                    # if genome.rules['rule_2a'][0] == 0:
                        # wipe out the one stick row
                    # pile = [i for i, x in enumerate(nim.rows) if x == 1][0]
                    # return Nimply(pile, 1)
                    # second row
                    # elif genome.rules['rule_2a'][0] == 1:
                    #     pile = random.choice([i for i, x in enumerate(nim.rows) if x >= 1])
                    #     num_objects_to_remove = nim.rows[pile] - genome.rules['rule_2a'][1]
                    #     if num_objects_to_remove < 1:
                    #         # take at least 1 stick (avoid weird negative numbers)
                    #         num_objects_to_remove = 1
                    #     return Nimply(pile, num_objects_to_remove)
                # else:
                #     print(nim.rows)
                #     print('here')
                #     pile = genome.rules['rule_2b'][0]
                #     num_objects_to_remove = nim.rows[pile] - genome.rules['rule_2b'][1]
                #     print(pile, num_objects_to_remove)
                #     return Nimply(pile, num_objects_to_remove)
                # rule 2b
                # else:
                #     if genome.rules['rule_2b'][0] == 0:
                #         row = nim.rows.index(stats['shortest_row'])
                #         print('shortest row', row)
                #     else:
                #         row = nim.rows.index(stats['longest_row'])
                #         print('longest row', row)
                #     num_objects_to_remove = max(nim.rows[row] - genome.rules['rule_2b'][1], 1)
                #     print('num_objects_to_remove', num_objects_to_remove)
                #     return Nimply(row, num_objects_to_remove)
            elif stats['num_active_rows'] >= 3:
                print('HERE')
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
                    print(nim.rows)
                    return Nimply(stats['longest_row'], max(max(nim.rows) - nim.rows[stats['shortest_row']], 1))
                else:
                    # do something random
                    return Nimply(*random.choice(stats['possible_moves']))

            else:
                # this is a fixed rule, does not have random component
                # rule from the paper Ryan Julian: The Game of Nim
                # If n piles and n-1 piles have the same size, remove x sticks from the smallest pile until it is the same size as the other piles
                # check if only 1 pile has a different number of sticks
                counter = Counter()
                for element in nim.rows:
                    counter[element] += 1
                if len(counter) == 2:
                    if counter.most_common()[0][1] == 1:
                        # remove x sticks from the smallest pile until it is the same size as the other piles
                        return Nimply(stats['shortest_row'], max(nim.rows[stats['shortest_row']] - counter.most_common()[1][0], 1))
                        # return Nimply(nim.rows.index(counter.most_common()[1][0]), 1)

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
        for i in range(5):
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
        return wins / 5

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