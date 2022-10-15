from collections import deque
import itertools
import random

class Graph:
    def __init__(self, adjacency_list, list_values, N):
        self.adjacency_list = adjacency_list
        self.list_values = list_values
        H = {}
        for key in list_values:
            # heuristic value is length of list
            H[key] = len(list_values[key])
        self.H = H
        print(self.H)
        # holds the lists of each visited node
        self.final_list = []
        # N is the count of elements that should be in the final list
        self.N = N

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        print(f"Heuristic value of {n}: {self.H[n]}")
        return self.H[n]

    def are_we_done(self):
        flattened_list = list(itertools.chain.from_iterable(self.final_list))
        for i in range(self.N):
            print(i)
            if i not in flattened_list:
                return False
        print("We are done")
        return True

    def a_star_algorithm(self, start_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

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

            self.final_list.append(self.list_values[n])

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if self.are_we_done():
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print(self.final_list)
                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                values = self.list_values[m]
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    print("Visiting neighbor: {}".format(m))
                    print("here")
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                    print(parents)

                # otherwise, check if it's more effective to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] + self.h(m) > g[n] + self.h(n) + weight:
                        print("Inside here")
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print(self.final_list)
        print('Path does not exist!')
        return None

list_values = {
    'A': [0, 1],
    'B': [2, 3],
    'C': [2, 4],
    'D': [4]
}

adjacency_list = {
    'A': [('B', len(list_values['B'])), ('C', len(list_values['C'])), ('D', len(list_values['D']))],
    'B': [('A', len(list_values['A'])), ('C', len(list_values['C'])), ('D', len(list_values['D']))],
    'C': [('A', len(list_values['A'])), ('B', len(list_values['B'])), ('D', len(list_values['D']))],
    'D': [('A', len(list_values['A'])), ('B', len(list_values['B'])), ('C', len(list_values['C']))]
}
graph1 = Graph(adjacency_list, list_values, N=4)
graph1.a_star_algorithm('A')

def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]

lists = problem(5, seed=20)
print(lists)
# conver to adjacency list
# adjacency_list = {}
# for i in range(len(lists)):
#     adjacency_list[i] = []
#     for j in range(len(lists)):
#         if i != j:
#             if len(set(lists[i]) & set(lists[j])) > 0:
#                 adjacency_list[i].append((j, 1))

# print(adjacency_list)
