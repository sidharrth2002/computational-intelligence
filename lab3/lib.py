from collections import namedtuple

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

class Genome:
    def __init__(self, rules):
        self.rules = rules
        self.fitness = 0

Nimply = namedtuple("Nimply", "row, num_objects")
