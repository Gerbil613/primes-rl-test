from copy import deepcopy

class Path:
    def __init__(self, s, r):
        self.reward = r
        self.states = deepcopy(s)

    def add(self, s, r):
        self.reward += r
        self.states.append(s)

    def __gt__(self, other):
        return self.reward > other.reward

    def __contains__(self, state):
        return state in self.states

    def __iter__(self):
        for state in self.states:
            yield state

    def __str__(self):
        return str(self.reward) + ': ' + str(self.states)