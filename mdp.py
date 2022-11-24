import numpy as np
from path import Path
from copy import deepcopy

class MDP:
    '''generalized MDP object
    assumes rewards are sampled from normal distribution and are deterministic with respect to new state
    terminals are INCLUSIVE; is a subset of state space
    states and actions are just indices, whereas state_space and action_space are the actual states and actions'''
    start = None
    score_means = []
    traversal_factors = {}
    paths = []
    P_star = None
    reward_deviations = None
    transition_function = None
    
    states = []
    actions = []

    def __init__(self, string='NOT_MAZE'):
        '''MDP(str) -> MDP
        initializes MDP, loads representation of maze if specified by directory parameter'''
        if string == 'NOT_MAZE': return # blank constructor

        self.state_space = []
        self.action_space = [[1,0],[0,1],[-1,0],[0,-1]]
        self.actions = range(4)

        grid = []
        with open(string, 'r') as maze_file:
            for line in maze_file.readlines():
                line = line.strip('\n').split(' ')
                grid.append(line)

        height, width = len(grid), len(grid[0])

        row = 0
        for line in grid:
            column = 0
            for item in line:
                id = 0 if len(self.states) == 0 else self.states[-1] + 1

                if item == 's': # start state
                    self.start = id
                    self.state_space.append([row, column]) # add actual state to state space
                    self.states.append(id) # add state index as well
                    self.score_means.append(0) # no reward for terminal state itself

                elif item.isdigit(): # regular state
                    self.state_space.append([row, column])
                    self.states.append(id)
                    self.score_means.append(int(item))
                
                column += 1
            row += 1

        self.transition_function = np.zeros((len(self.states), 4, len(self.states)))
        self.reward_deviations = np.zeros((len(self.states))) + 1

        def recursive_maze_load(self, state, visited, current_path, factor, width, height):
            '''MDP.recursive_maze_load(state, set, arr, int) -> None
            a) recursively finds every path in maze and stores each one's constituent states and total reward sum
            b) calculates and stores traversal factor of every node
            c) sets up transition function'''
            self.traversal_factors[state] = factor
            visited.add(state)

            current_path.states.append(state) # add current state to path
            current_path.reward += self.score_means[state] # add reward to running total

            new_factor = 0
            for action in self.action_space:
                r, c = self.state_space[state]
                try:
                    new_state =  self.state_space.index([r + action[0], c + action[1]])
                    
                except ValueError: continue

                if new_state not in visited:
                    new_factor += 1

            if new_factor == 0:
                self.paths.append(current_path)
                return

            else:
                new_factor = factor * 1/float(new_factor)

            for action in self.action_space: # try each action and see which next states we can go to according to TF
                r, c = self.state_space[state]
                try:
                    new_state =  self.state_space.index([r + action[0], c + action[1]])

                except ValueError: continue

                if new_state not in visited:
                    self.transition_function[state][self.action_space.index(action)][new_state] = 1
                    recursive_maze_load(self, new_state, visited, deepcopy(current_path), new_factor, width, height)

        recursive_maze_load(self, self.start, set(), Path([], 0), 1, width, height) # set up TF et al. for maze
        self.paths.sort(reverse=True) # best is first
        self.P_star = self.paths[0]

    def take_action(self, state, action):
        '''MDP.take_action(state, action) -> real number, state
        inputs current state and action to take, and outputs new state and reward acquired in the process
        this is transitions dynamic function'''
        new_state = self.sample_transition_function(state, action)
        reward = self.sample_reward_function(state, action)
        return [reward, new_state]

    def sample_transition_function(self, state, action):
        '''MDP.sample_transition_function(state, action) -> state
        given state and action, finds new state based on transition function's probability distribution'''
        distribution = self.transition_function[state][action]
        return np.random.choice(self.states, p=distribution)

    def sample_reward_function(self, state, action):
        '''MDP.sample_reward_function(state, action) -> real number
        given state and action, samples reward function'''
        new_state = self.sample_transition_function(state, action)
        return np.random.normal(loc=self.score_means[new_state], scale=self.reward_deviations[new_state])

    def get_action_space(self, state):
        '''MDP.get_action_space(state) -> arr of actions
        outputs set up actions that may be taken in given state'''
        output = []
        for action in self.actions:
            if np.any(np.greater(self.transition_function[state][action], 0)): # if there are any nonzeros for this action
                output.append(action)

        return output

