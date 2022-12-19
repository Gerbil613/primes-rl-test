from copy import deepcopy

class Path:
    def __init__(self, s, r):
        self.reward = r
        self.states = deepcopy(s)
        self.id = -1

    def add(self, s, r):
        self.reward += r
        self.states.append(s)

    def __gt__(self, other):
        return self.reward > other.reward

    def __contains__(self, edge):
        if edge[0] in self.states and edge[1] in self.states:
            return self.states.index(edge[0]) == self.states.index(edge[1]) - 1
        return False

    def __iter__(self):
        for i in range(len(self.states) - 1):
            yield (self.states[i], self.states[i+1])

    def __str__(self):
        return str(self.reward) + ': ' + str(self.states)