from venv import create
import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from math_functions import *

blocks = [] # list of invalid states (walls)
terminals = [] # list of terminal states
start = None # state at which we start
scores = {} # dict maps the hash of a state to the reward associated with it
transition_function = None # just declare
corrupted_transition_function = None
width, height = 6, 3

actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)
values = np.zeros((height, width, 4)) # will be initialized as np.array of shape (height, width), outputs value

def hash(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return a * width + b

def unhash(hash):
    column = hash % width
    row = int((hash - column) / width)
    return [row, column]

with open('simplest_maze.txt', 'r') as maze_file:
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

def create_corrupted_transition(worst_case=False, attack_strength=0):
    '''create_corrupted_transition() -> arr
    adds some noise to transition function'''
    output = deepcopy(initial_transition_function)
    if not worst_case:
        for state in scores:
            state = unhash(state)
            action_space = get_action_space(state)
            if len(action_space) >= 2:
                poss_new_states = [[state[0] + action [0], state[1] + action[1]] for action in action_space]
                for action in action_space:
                    for new_state in poss_new_states:
                        output[hash(*state)][actions.index(action)][hash(*new_state)] += np.random.normal(
                            loc = 0.0,
                            scale = attack_strength
                        )
                    output[hash(*state)][actions.index(action)] -= np.min(output[hash(*state)][actions.index(action)])
                    output[hash(*state)][actions.index(action)] /= np.sum(output[hash(*state)][actions.index(action)])

        return output
    
    output[10][3][9] = 1 - attack_strength
    output[10][3][11] = attack_strength
    output[10][1][11] = 1 - attack_strength
    output[10][1][9] = attack_strength

    return output

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

    transition_function = corrupted_transition_function

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

def main():
    # DO NOT DELETE THESE FOUR LINES; they're essential
    global initial_transition_function, transition_function, corrupted_transition_function
    initial_transition_function = create_initial_transition()
    transition_function = initial_transition_function
    corrupted_transition_function = create_corrupted_transition(worst_case=False, attack_strength=0)

    data = []
    num_trials = 20
    num_sub_trials = 20
    attack_strength = 0
    for trial in range(num_trials):
        print('Trial', trial+1, attack_strength)
        corrupted_transition_function = create_corrupted_transition(worst_case=True, attack_strength=attack_strength)
        avg = 0
        for sub_trial in range(num_sub_trials):
            learn(2000, 0.99, 0.7, 0.1, defend=True, verbose=False)
            avg += evaluate()
        
        data.append(avg / float(num_sub_trials))
        attack_strength += 1 / float(num_trials)

    print(data)

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamic function'''
    new_state = sample_transition_function(state, action)
    reward = get_reward(new_state)
    return [reward, new_state]

def sample_transition_function(state, action):
    '''very sus sampling function to transition to next state according to the transition matrix probabilities and current state+action'''
    '''basically we loop thru the matrix for the column w/ fixed current state and action and add up the probabilities and see if our chosen random number falls under this range'''
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