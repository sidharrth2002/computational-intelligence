from collections import deque
import itertools
import random
import time

class Graph:
    def __init__(self, adjacency_list, list_values, N):
        self.adjacency_list = adjacency_list
        self.list_values = list_values
        H = {}
        for key in list_values:
            # heuristic value is length of list
            H[key] = len(list_values[key])
        self.H = H
        # holds the lists of each visited node
        self.final_list = []
        # N is the count of elements that should be in the final list
        self.N = N
        self.discovered_elements = set()

    def flatten_list(self, _list):
        return list(itertools.chain.from_iterable(_list))

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def get_number_of_elements_not_in_second_list(self, list1, list2):
        count = 0
        # flattened_list = self.flatten_list(list2)
        for i in list1:
            if i not in list2:
                count += 1
        return count

    # f(n) = h(n) + g(n)

    def h(self, n):
        num_new_elements = self.get_number_of_elements_not_in_second_list(self.list_values[n], self.discovered_elements)
        if self.list_values[n] in self.final_list:
            return 100
        return self.H[n] - num_new_elements
        # return self.H[n] / (num_new_elements + 1)

    def get_node_with_least_h(self):
        min_h = float("inf")
        min_node = None
        for node in self.adjacency_list:
            if self.h(node) < min_h:
                min_h = self.h(node)
                min_node = node
        return min_node

    # visited_node = [1, 2, 3]
    # final_list = [[4, 5], [1]]
    def are_we_done(self):
        # flattened_list = list(itertools.chain.from_iterable(self.final_list))
        for i in range(self.N):
            if i not in self.discovered_elements:
                return False
        print("We are done")
        return True

    def insert_unique_element_into_list(self, _list, element):
        if element not in _list:
            _list.append(element)
        return _list

    def a_star_algorithm(self):
        # start_node is node with lowest cost
        start_node = self.get_node_with_least_h()

        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        # open_list = [set([start_node])]
        # closed_list = set([])
        open_list = [start_node]
        closed_list = []

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;

            if n == None:
                print('Path does not exist!')
                return None

            print(f"Visiting node: {n}")
            self.final_list.append(self.list_values[n])
            # self.discovered_elements.union(self.list_values[n])
            # add list_values[n] to discovered_elements
            for i in self.list_values[n]:
                self.discovered_elements.add(i)
            print(len(self.discovered_elements))

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if self.are_we_done():
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print(f"Number of elements in final list: {len(self.flatten_list(self.final_list))}")
                print('Path found: {}'.format(reconst_path))
                print(
                    f"I guess better looking solution for N={N}: w={sum(len(_) for _ in self.final_list)} (bloat={(sum(len(_) for _ in self.final_list)-N)/N*100:.0f}%)"
                )
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                values = self.list_values[m]
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    # open_list.add(m)
                    open_list = self.insert_unique_element_into_list(open_list, m)
                    # sort open_list by self.h
                    open_list = sorted(open_list, key=self.h)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's more effective to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] + self.h(m) > g[n] + self.h(n) + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            # open_list.add(m)
                            open_list = self.insert_unique_element_into_list(open_list, m)
                            open_list = sorted(open_list, key=self.h)


            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            open_list = sorted(open_list, key=self.h)
            # closed_list.add(n)
            closed_list = self.insert_unique_element_into_list(closed_list, n)

        print('Path does not exist!')
        return None

def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]

# list_of_lists = [[3], [0, 6], [1, 7], [3], [1], [5, 6], [6], [3], [1, 4, 6], [7], [2, 4], [1, 2, 3], [0, 1, 6], [2, 7], [5, 7], [0], [7], [4, 7], [2], [3, 6], [4, 5, 6], [4, 5, 7], [2, 4], [0, 1, 3], [0, 4, 6], [7]]
N = 1000
list_of_lists = problem(N, 42)
print(max(list(itertools.chain.from_iterable(list_of_lists))))

list_values = {}
for i in range(len(list_of_lists)):
    list_values[str(i)] = list_of_lists[i]

adjacency_list = {}
for i in range(len(list_of_lists)):
    adjacency_list[str(i)] = []
    for j in range(len(list_of_lists)):
        if i != j:
            adjacency_list[str(i)].append((str(j), 1))

# list_values = {
#     'A': [0, 1],
#     'B': [2, 3],
#     'C': [2, 4],
#     'D': [4]
# }

# adjacency_list = {
#     'A': [('B', len(list_values['B'])), ('C', len(list_values['C'])), ('D', len(list_values['D']))],
#     'B': [('A', len(list_values['A'])), ('C', len(list_values['C'])), ('D', len(list_values['D']))],
#     'C': [('A', len(list_values['A'])), ('B', len(list_values['B'])), ('D', len(list_values['D']))],
#     'D': [('A', len(list_values['A'])), ('B', len(list_values['B'])), ('C', len(list_values['C']))]
# }
graph1 = Graph(adjacency_list, list_values, N=N)

start_time = time.time()
graph1.a_star_algorithm()
end_time = time.time()
print(f"Time taken: {end_time - start_time}")

# def problem(N, seed=None):
#     random.seed(seed)
#     return [
#         list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
#         for n in range(random.randint(N, N * 5))
#     ]

def flatten_list(_list):
    return list(itertools.chain.from_iterable(_list))

def greedy(N):
    goal = set(range(N))
    covered = set()
    solution = list()
    all_lists = sorted(problem(N, seed=42), key=lambda l: len(l))
    while goal != covered:
        x = all_lists.pop(0)
        if not set(x) < covered:
            solution.append(x)
            covered |= set(x)
    print(f"Number of elements in final list: {len(flatten_list(solution))}")
    print(
        f"Greedy solution for N={N}: w={sum(len(_) for _ in solution)} (bloat={(sum(len(_) for _ in solution)-N)/N*100:.0f}%)"
    )

print("Greedy")
start_time = time.time()
greedy(N)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")

lists = problem(8, seed=42)
# conver to adjacency list
# adjacency_list = {}
# for i in range(len(lists)):
#     adjacency_list[i] = []
#     for j in range(len(lists)):
#         if i != j:
#             if len(set(lists[i]) & set(lists[j])) > 0:
#                 adjacency_list[i].append((j, 1))

# print(adjacency_list)
