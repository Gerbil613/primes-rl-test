import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from math_functions import *
from path import Path

blocks = [] # list of invalid states (walls)
terminals = [] # list of terminal states
start = None # state at which we start
scores = {} # dict maps the hash of a state to the reward associated with it
transition_function = None # just declare
width, height = 6, 3

p = 0.25
delta = 4

actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)
values = np.zeros((height, width, 4)) # will be initialized as np.array of shape (height, width), outputs value

paths = [] # stores array of Path objects
P_star = None

def hash(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return a * width + b

def unhash(hash):
    column = hash % width
    row = int((hash - column) / width)
    return [row, column]

with open('testmaze.txt', 'r') as maze_file:
    row = 0
    for line in maze_file.readlines():
        line = line.strip('\n').split(' ')
        column = 0
        for item in line:
            if item == '|': # wall
                blocks.append(hash(row, column))
            elif item == '*': # terminal state
                terminals.append(hash(row, column))
                scores[hash(row, column)] = 0 # no reward for start
            elif item == 's': # start state
                start = [row, column]
                scores[hash(row, column)] = 0 # no reward for terminal state itself

            else: # regular state
                scores[hash(row, column)] = int(item)
            
            column += 1

        width = column
        row += 1

    height = row

def get_degree(state):
    '''get_degree(int) -> float
    outputs number of paths through state; return float because output is used as denominator'''
    count = 0.0
    for path in paths:
        if state in path: count += 1

    return count
    #return float(np.sum(np.any(transition_function[state], axis=1))) # compute how many states you can transitino to from this state

def determine_path_to_corrupt():
    P_p = P_star
    for P_i in paths: # don't iterate over best path
        if P_i.reward == P_star.reward: continue
        b_i = 2
        for P_j in paths:
            if P_j.reward == P_i.reward or P_j.reward == P_star.reward: continue

            deg_star = -1 # just need to initialize to something, is really infinity though
            for state in P_j:
                if (state in P_i) != (state in P_star) and (get_degree(state) < deg_star or deg_star == -1):
                    deg_star = get_degree(state)
            
            if deg_star != -1: b_i += 1/deg_star

        if P_i.reward < P_p.reward and P_star.reward - b_i*p*delta < P_i.reward:
            P_p = P_i

    return P_p

def get_paths_recursive(r, c, visited, current_path):
    '''get_paths_recursive(int, int, set, arr, int) -> None
    recursively finds every path in MDP and stores each one's constituent states and total reward sum'''
    if hash(r,c) in terminals: # reached end of path
        paths.append(current_path) # store the data 
        return

    visited.add(hash(r,c))

    current_path.states.append(hash(r,c)) # add current state to path
    current_path.reward += scores[hash(r, c)] # add reward to running total

    for action in actions: # try each action and see which next states we can go to according to TF
        for new_state_hashed, probability in enumerate(transition_function[hash(r,c)][actions.index(action)]):
            new_r, new_c = unhash(new_state_hashed)[0], unhash(new_state_hashed)[1]
            if probability > 0 and hash(new_r, new_c) not in visited: # this state is viable next option
                get_paths_recursive(new_r, new_c, visited, deepcopy(current_path))

def create_initial_transition():
    '''creates a new instance of default transition function (default setting is original deterministic)'''
    output = np.zeros(shape=(width*height, 4, width*height))

    for state in range(width*height):
        for new_state in range(width*height):
            for action_ind in range(4):
                action = actions[action_ind]
                row, col, new_row, new_col = *unhash(state), *unhash(new_state)
                if state not in blocks and new_state not in blocks and row + action[0] == new_row and col + action[1] == new_col:
                    output[state][action_ind][new_state] = 1

    stop_backtracking(*start, set(), output)
    return output

def stop_backtracking(row, column, visited, transition_function):
    '''stop_backtracking(int, int, int, arr) -> None
    recursive function that sets the inputted transition function so that you can't go backwards'''
    if hash(row, column) in terminals:
        return

    visited.add(hash(row, column))
    for action in actions:
        new_state = [row + action[0], column + action[1]]
        if new_state[0] >= 0 and new_state[0] < height and new_state[1] >= 0 and new_state[1] < width:
            
            if hash(*new_state) in scores and hash(*new_state) not in visited:
                transition_function[hash(*new_state)][actions.index([-1*action[0], -1*action[1]])][hash(row, column)] = 0
                stop_backtracking(*new_state, visited, transition_function)

def get_policy(values, epsilon):
    '''get_policy(arr, float) -> arr
    outputs the agent's actual policy distribution using softmax on q-values'''
    policy = np.zeros((height, width, 4))
    for row in range(height):
        for column in range(width):
            policy[row][column] = np.exp(values[row][column]) / np.sum(np.exp(values[row][column]))

    return policy

def learn(num_episodes, gamma, epsilon, alpha, defend=False, verbose=False):
    '''learn(int, float, float, float, bool) -> None
    trains global variable "values" to learn Q-values of maze'''
    global visited, values, transition_function

    values = np.zeros((height, width, 4)) # intialize
    for episode in range(num_episodes):
        state = start
        if verbose:
            if episode % 100 == 0: print(episode)
        while not hash(*state) in terminals: # so long as we don't hit an end
            action = best_action(state, epsilon) # get best action according to current policy
            reward, new_state = take_action(state, action) # take action and observe reward, new state
            if defend: reward = reward + entropy(get_policy(values, epsilon)[state[0]][state[1]]) # maxent transformation
            values[state[0]][state[1]][actions.index(action)] = \
                 (1-alpha) * get_value(state, action) + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
            state = new_state

def evaluate():
    '''evaluate() -> int
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    global transition_function
    performance = 0
    state = start
    transition_function = initial_transition_function
    # simply loop and keep on using policy to progress through maze
    while not hash(*state) in terminals:
        reward, new_state = take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    #print('Ended evaluation at: ' + str(state))
    return performance

def init():
    '''init() -> None
    general variable initiation protocol; sets up transition function and stores all paths in MDP'''
    global initial_transition_function, transition_function, P_star
    initial_transition_function = create_initial_transition()
    transition_function = initial_transition_function

    get_paths_recursive(*start, set(), Path([], 0))
    paths.sort(reverse=True) # best is first
    P_star = paths[0]

def main():
    init()
    switch_path = determine_path_to_corrupt()
    print('Switch optimal path with path of reward ' + str(switch_path.reward))
    '''learn(20000, 0.99, 0.7, 0.3, verbose=True)
    print(evaluate())
    plt.imshow(np.sum(values, axis=2))
    plt.show()'''

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamic function'''
    new_state = sample_transition_function(state, action)
    reward = get_reward(new_state)
    return [reward, new_state]

def sample_transition_function(state, action):
    distribution = transition_function[hash(*state)][actions.index(action)]
    return unhash(np.random.choice(range(width*height), p=distribution))
  
def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    return scores[hash(new_state[0],new_state[1])]

def get_value(state, action):
    '''get_value(tuple, tuple) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    if action == [0,0]: return 0 # this happens when we get the value at a terminal state; the computer tries the dummy action [0,0] at terminals
    return values[state[0]][state[1]][actions.index(action)]

def get_action_space(state):
    ''''get_action_space(tuple) -> arr
    outputs list of all possible actions that may be taken in state'''
    global transition_function
    if hash(*state) in terminals: return []
    output = []
    for action in actions:
        if np.any(np.greater(transition_function[hash(*state)][actions.index(action)], 0)): # if there are any nonzeros for this action
            output.append(action)

    return output

def best_action(state, epsilon):
    '''best_action(tuple, float) -> tuple
    outputs the best action in the current state according to current value function, but takes random action with probability epsilon'''
    poss_actions = get_action_space(state)
    #if state == [2,3]: print(poss_actions)
   
    if random.random() < epsilon:
        return random.choice(poss_actions)
    if len(poss_actions) == 0:
        return [0,0]

    best_action, best_value = poss_actions[0], get_value(state, poss_actions[0])
    for action in poss_actions:
        if get_value(state, action) > best_value:
            best_value = get_value(state, action)
            best_action = action
    return best_action

if __name__ == '__main__':
    main()