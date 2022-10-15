import random
# Given a number and some lists of integers, find a list of lists where each number between 0 and N-1 appears in at least one list.

class Node:
    def __init__(self, value, parent=None):
        # value is a list
        self.value = value
        # cost is the length of the list
        self.cost = len(value)

def a_star(start, goal):
    start_node = Node(start)


# def minimum_list(N, lists):
#     new_list = []
#     for i in range(len(lists)):
#         for j in range(N):
#             if j in lists[i]:
#                 new_list.append(lists[i])
#     return new_list

def problem(N, seed=None):
    random.seed(seed)
    return [
        list(set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2))))
        for n in range(random.randint(N, N * 5))
    ]



# min_list = minimum_list(10, problem(10, 0))
# print(min_list)


