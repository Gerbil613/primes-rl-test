import numpy as np
from path import Path
from copy import deepcopy
import pickle

class MDP:
    '''generalized MDP object
    assumes rewards are sampled from normal distribution and are deterministic with respect to new state
    terminals are INCLUSIVE; is a subset of state space
    states and actions are just indices, whereas state_space and action_space are the actual states and actions'''
    start = None
    rewards = []
    traversal_factors = None
    paths = []
    P_star = None
    transition_function = None
    visited = set()
    unique_edges = [] # list which maps a path to the edge unique to it
    
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
        self.raw_scores = [] # this stores rewards with respect to new_state; will be reprocessed later to be in state, new_state format

        row = 0
        for line in grid:
            column = 0
            for item in line:
                id = 0 if len(self.states) == 0 else self.states[-1] + 1

                if item == 's': # start state
                    self.start = id
                    self.state_space.append([row, column]) # add actual state to state space
                    self.states.append(id) # add state index as well
                    self.raw_scores.append(0) # no reward for start state itself

                elif item.isdigit(): # regular state
                    self.state_space.append([row, column])
                    self.states.append(id)
                    self.raw_scores.append(int(item))
                
                column += 1
            row += 1

        self.transition_function = np.zeros((len(self.states), 4, len(self.states)))
        self.rewards = np.zeros((len(self.states), len(self.states)))

        def load_maze_tf(self, state, visited, width, height):
            '''load_maze_tf(state, set, int) -> None
            sets up transition funciton and loads paths and traversal factors'''
            visited.add(state)

            for action in self.action_space: # try each action and see which next states we can go to according to TF
                r, c = self.state_space[state]
                try: new_state =  self.state_space.index([r + action[0], c + action[1]])
                except ValueError: continue

                if new_state not in visited:
                    self.transition_function[state][self.action_space.index(action)][new_state] = 1
                    self.rewards[state, new_state] = self.raw_scores[new_state] # reprocess to be in right format
                    load_maze_tf(self, new_state, visited, width, height)

        load_maze_tf(self, self.start, set(), width, height) # set up TF etc. for maze
        self.load_paths(self.start, Path([], 0))
        self.paths.sort(reverse=True) # best is first
        for id in range(len(self.paths)): self.paths[id].id = id
        self.P_star = self.paths[0]
        self.load_traversal_factors()

    def load_random_dag(self, num_states, p_edge=.55, reward_std=1, assure_unique_edges=True):
        '''MDP.load_random_dag(int) -> None
        loads random DAG MDP with specified number of states'''
        self.states = list(range(num_states))
        self.start = 0
        while True:
            self.visited = set()
            self.paths = []
            self.rewards = np.zeros((num_states, num_states))
            num_actions = 0
            self.transition_function = np.zeros((num_states, num_states - 1, num_states))
            for state1 in range(num_states):
                action = 0
                for state2 in range(state1+1, num_states):
                    if np.random.random() < p_edge:
                        self.rewards[state1, state2] = np.random.normal(loc=0, scale=reward_std)
                        self.transition_function[state1][deepcopy(action)][state2] = 1
                        action += 1

                num_actions = max(action, num_actions)

            # now we test to make sure that graph is fully connected
            self.transition_function = np.delete(self.transition_function, np.s_[num_actions:], axis=1) # truncate transition function based on how many actions there are
            self.actions = list(range(num_actions))
            self.load_paths(self.start, Path([], 0))
            if len(self.visited) == len(self.states) and len(self.paths) > 1: break
        
        self.paths.sort(reverse=True) # best is first
        self.P_star = self.paths[0]
        for id in range(len(self.paths)): self.paths[id].id = id
        self.load_traversal_factors()
        if assure_unique_edges: self.assure_unique_edges(reward_std)

    def load_random_layered(self, num_layers, mean_nodes_per_layer, mean_degree, reward_std, assure_unique_edges=True):
        '''MDP.load_random_layered(int, int, int, float, bool=True)
        load randomly generated layered MDP, which is a subset of DAG'''
        self.states = []
        self.actions = []
        self.start = 0
        self.num_layers = num_layers
        if mean_nodes_per_layer < mean_degree: raise ValueError('Mean degree cannot be greater than mean number of nodes per layer.')
        min_num_nodes_per_layer = mean_degree
        max_num_nodes_per_layer = 2*mean_nodes_per_layer - mean_degree
        node_counts = np.random.randint(min_num_nodes_per_layer, max_num_nodes_per_layer + 1, size=num_layers) # +1 cuz exclusive
        node_counts[0] = 1 # there is one start node in the first layer
        num_states = np.sum(node_counts)
        current_state = 0
        structure = []
        for node_count in node_counts:
            current_arr = []
            for i in range(node_count):
                current_arr.append(current_state)
                current_state += 1

            structure.append(current_arr)

        assert current_state == num_states

        self.states = list(range(num_states))
        self.visited = set()
        self.paths = []
        self.rewards = np.zeros((num_states, num_states))
        num_actions = 0
        self.transition_function = np.zeros((num_states, num_states - 1, num_states))
        for layer_index in range(num_layers - 1):
            # connnect from layer_index to layer_index + 1
            while True:
                connected = set()
                edges_drawn = set()
                for state in structure[layer_index]:
                    action = 0
                    for new_state in structure[layer_index + 1]:
                        probability = float(mean_degree) / mean_nodes_per_layer
                        if np.random.random() < probability:
                            connected.add(new_state)
                            edges_drawn.add((state, new_state))
                            self.transition_function[state][deepcopy(action)][deepcopy(new_state)] = 1
                            self.rewards[state][new_state] = np.random.normal(loc=0, scale=reward_std)
                            action += 1
                            num_actions = max(num_actions, action)

                if len(connected) == len(structure[layer_index + 1]):
                    break # keep going till next layer is fully connected
                
                else:
                    for edge in edges_drawn: # reset TF and rewards!
                        state, new_state = edge
                        self.transition_function[state, :, new_state] = np.zeros((num_states - 1,))
                        self.rewards[state][new_state] = 0

        self.transition_function = np.delete(self.transition_function, np.s_[num_actions:], axis=1) # truncate transition function based on how many actions there are
        self.actions = list(range(num_actions))
        self.load_paths(self.start, Path([], 0))        
        self.paths.sort(reverse=True) # best is first
        self.P_star = self.paths[0]
        for id in range(len(self.paths)): self.paths[id].id = id
        self.load_traversal_factors()
        if assure_unique_edges: self.assure_unique_edges(reward_std)

    def load_paths(self, state, current_path, prev_state=-1):
        '''MDP.load_paths(state, path, prev_state) -> None
        loads paths in MDP based on TF'''
        current_path.states.append(state)
        self.visited.add(state)
        if prev_state != -1: current_path.reward += self.rewards[prev_state, state]
        num_actions = np.sum(self.transition_function[state])
        if num_actions == 0:
            self.paths.append(current_path)
            return

        for new_state in self.states:
            for action in self.actions:
                if self.transition_function[state][action][new_state] == 1: # we can get to new state
                    self.load_paths(new_state, deepcopy(current_path), prev_state=state)

    def load_traversal_factors(self):
        '''MDP.load_traversal_factors() -> None
        once MDP is set up, calculates and stores traversal factors'''
        self.traversal_factors = np.zeros((len(self.states), len(self.states)))
        for state1 in self.states:
            for state2 in self.states:
                if not self.transition_function[state1, :, state2].any(): continue
            
                edge = (state1, state2)
                for path in self.paths:
                    if edge in path:
                        self.traversal_factors[state1, state2] += 1

    def assure_unique_edges(self, reward_std):
        '''MDP.assure_unique_edges(float) -> None
        assures that every path in the MDP has an edge unique to that path'''
        self.unique_edges = [None] * len(self.paths)
        for path in self.paths:
            has_unique_edge = False
            for edge in path:
                if self.traversal_factors[edge] == 1:
                    self.unique_edges[path.id] = edge
                    has_unique_edge = True
            
            if not has_unique_edge:
                tip_state = len(self.states)
                self.states.append(tip_state)

                self.rewards = np.append(self.rewards, np.zeros((len(self.rewards), 1)), axis=1) # lengthen reward function for tip states
                reward = np.random.normal(loc=0, scale=reward_std)
                self.rewards[path.states[-2], tip_state] = reward # -2 because -1 is the tip_state itself, since we added it already

                self.transition_function = np.append(self.transition_function, np.zeros((len(self.transition_function), len(self.transition_function[0]), 1)), axis=2)
                self.transition_function[path.states[-2], 0, tip_state] = 1
                path.add(tip_state, reward)

                self.traversal_factors = np.append(self.traversal_factors, np.zeros((len(self.traversal_factors), 1)), axis=1)
                self.traversal_factors[path.states[-2], tip_state] = 1 # unique

                self.unique_edges[path.id] = [path.states[-2], tip_state]

    def take_action(self, state, action):
        '''MDP.take_action(state, action) -> real number, state
        inputs current state and action to take, and outputs new state and reward acquired in the process
        this is transitions dynamic function'''
        new_state = self.sample_transition_function(state, action)
        reward = self.sample_reward_function(state, new_state)
        return [reward, new_state]

    def sample_transition_function(self, state, action):
        '''MDP.sample_transition_function(state, action) -> state
        given state and action, finds new state based on transition function's probability distribution'''
        distribution = self.transition_function[state][action]
        return np.random.choice(self.states, p=distribution)

    def sample_reward_function(self, state, new_state):
        '''MDP.sample_reward_function(state, new_state) -> real number
        given state and action, samples reward function'''
        reward = self.rewards[state, new_state]
        return reward

    def get_action_space(self, state):
        '''MDP.get_action_space(state) -> arr of actions
        outputs set up actions that may be taken in given state'''
        output = []
        for action in self.actions:
            if np.any(np.greater(self.transition_function[state][action], 0)): # if there are any nonzeros for this action
                output.append(action)

        return output

    def get_average_depth(self):
        '''MDP.get_average_depth() -> float
        outputs mean depth over all paths in MDP'''
        return np.average([len(path.states) for path in self.paths])

    def __str__(self):
        '''str(MDP) -> str
        Outputs a string representation of the MDP'''
        return 'Number of States: ' + str(len(self.states)) + '\nNumber of actions: ' + str(len(self.actions)) \
        + '\nPaths: ' + str([str(path) for path in self.paths]) + '\nTransition function:\n' + str(self.transition_function) + '\nReward function:\n' + str(self.rewards)

