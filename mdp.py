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
    transition_function = None
    
    states = []
    actions = []

    def load_maze(self, string):
        '''MDP.load_maze(str) -> None
        loads representation of maze if specified by directory parameter'''
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
                    self.score_means.append(0) # no reward for start state itself

                elif item.isdigit(): # regular state
                    self.state_space.append([row, column])
                    self.states.append(id)
                    self.score_means.append(int(item))
                
                column += 1
            row += 1

        self.transition_function = np.zeros((len(self.states), 4, len(self.states)))

        def recursive_maze_load(self, state, visited, current_path, width, height):
            '''MDP.recursive_maze_load(state, set, arr, int) -> None
            a) recursively finds every path in maze and stores each one's constituent states and total reward sum
            b) calculates and stores traversal factor of every node
            c) sets up transition function'''
            visited.add(state)

            current_path.states.append(state) # add current state to path
            current_path.reward += self.score_means[state] # add reward to running total

            is_terminal = True
            for action in self.action_space: # try each action and see which next states we can go to according to TF
                r, c = self.state_space[state]
                try: new_state =  self.state_space.index([r + action[0], c + action[1]])
                except ValueError: continue

                if new_state not in visited:
                    self.transition_function[state][self.action_space.index(action)][new_state] = 1
                    is_terminal = False
                    recursive_maze_load(self, new_state, visited, deepcopy(current_path), width, height)

            if is_terminal:
                self.paths.append(current_path)
                return

        recursive_maze_load(self, self.start, set(), Path([], 0), width, height) # set up TF etc. for maze
        self.paths.sort(reverse=True) # best is first
        self.P_star = self.paths[0]
        self.load_traversal_factors()

    def load_random(self, num_states, p_edge=0.5):
        '''MDP.load_random(int) -> None
        loads random MDP with specified number of states'''
        self.states = range(num_states)
        self.start = 0
        matrix = np.zeros((num_states, num_states))
        for state1 in range(num_states):
            for state2 in range(state1+1, num_states):
                if np.random.random() < p_edge:
                    matrix[state1][state2] = 1
                    matrix[state2][state1] = 1

        self.score_means = np.zeros((num_states))

        # next piece of code connects everything up in matrix
        next_states = [self.start] # stack that does BFS
        unvisited = set(self.states)
        while len(next_states) > 0:
            state = next_states.pop(0)
            unvisited.remove(state)
            self.score_means[state] = np.random.random() * 10

            next_states = []
            for new_state in unvisited:
                if matrix[state][new_state] == 1: # there is an edge connecting these two states
                    next_states.append(new_state)

            if len(next_states) == 0 and len(unvisited) > 0: # all remaining states are fully disconnected from the current group of states, we need to "hop" over to it
                new_state = np.random.choice(list(unvisited))
                matrix[state][new_state] = 1
                next_states = [new_state]

        num_actions = int(np.max(np.sum(matrix, axis=0)))
        self.transition_function = np.zeros((num_states, num_actions, num_states))

        # this sets up transition function
        global_unvisited = deepcopy(list(self.states))
        def load_transition_function(state, local_unvisited):
            local_unvisited.remove(state)
            global_unvisited.remove(state)
            
            action = 0
            for new_state in local_unvisited:
                if matrix[state][new_state] == 1:
                    if new_state in global_unvisited: # never visited this state before
                        self.transition_function[state][action][new_state] = 1
                        load_transition_function(new_state, deepcopy(local_unvisited))
                        action += 1

                    elif new_state not in global_unvisited: # visited this new_state in different branch
                        self.transition_function[state][action][new_state] = 1
                        action += 1

                if action not in self.actions: self.actions.append(action)

        load_transition_function(self.start, deepcopy(list(self.states)))
        def load_paths_factors(state, unvisited, current_path, factor):
            unvisited.remove(state)
            self.traversal_factors[state] = factor
            current_path.states.append(state)
            current_path.reward += self.score_means[state]
            num_actions = np.sum(self.transition_function[state])
            if num_actions == 0:
                self.paths.append(current_path)
                return

            new_factor = factor / float(num_actions)
            for new_state in unvisited:
                for action in self.actions:
                    if self.transition_function[state][action][new_state] == 1: # we can get to new state
                        load_paths_factors(new_state, unvisited, current_path, new_factor)

        load_paths_factors(self.start, deepcopy(list(self.states)), Path([], 0), 1)
        self.paths.sort(reverse=True) # best is first
        self.P_star = self.paths[0]

    def load_traversal_factors(self):
        '''MDP.load_traversal_factors() -> None
        once MDP is set up, calculates and stores traversal factors'''
        self.traversal_factors = np.zeros((len(self.states)))
        for state in self.states:
            for path in self.paths:
                if state in path.states:
                    self.traversal_factors[state] += 1 / float(len(self.paths))

    def take_action(self, state, action, attack=0, delta=0, p=0, path_to_corrupt=None):
        '''MDP.take_action(state, action, int, float, path) -> real number, state
        inputs current state and action to take, and outputs new state and reward acquired in the process
        this is transitions dynamic function'''
        new_state = self.sample_transition_function(state, action)
        reward = self.sample_reward_function(state, action, attack=attack, delta=delta, p=p, path_to_corrupt=path_to_corrupt)
        return [reward, new_state]

    def sample_transition_function(self, state, action):
        '''MDP.sample_transition_function(state, action) -> state
        given state and action, finds new state based on transition function's probability distribution'''
        distribution = self.transition_function[state][action]
        return np.random.choice(self.states, p=distribution)

    def sample_reward_function(self, state, action, attack=0, delta=0, p=0, path_to_corrupt=None):
        '''MDP.sample_reward_function(state, action, int, float, path) -> real number
        given state and action, samples reward function'''
        new_state = self.sample_transition_function(state, action)
        reward = self.score_means[new_state]
        if attack == 0: return reward
        if attack == 1 and np.random.random() < p: return np.random.normal(loc=reward, scale=1)
        if attack == 2 and np.random.random() < p:
            if new_state in path_to_corrupt and new_state not in self.P_star:
                reward += delta

            elif new_state not in path_to_corrupt and new_state in self.P_star:
                reward -= delta

            return reward

    def get_action_space(self, state):
        '''MDP.get_action_space(state) -> arr of actions
        outputs set up actions that may be taken in given state'''
        output = []
        for action in self.actions:
            if np.any(np.greater(self.transition_function[state][action], 0)): # if there are any nonzeros for this action
                output.append(action)

        return output

