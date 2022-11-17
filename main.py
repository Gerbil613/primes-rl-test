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

RAND = 0
GREED = 1

p = 0.25
delta = 4

actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)
values = np.zeros((height, width, 4)) # will be initialized as np.array of shape (height, width), outputs value
traversal_factors = {}

paths = [] # stores array of Path objects
P_star = None
path_to_corrupt = None

def hash(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return a * width + b

def unhash(hash):
    column = hash % width
    row = int((hash - column) / width)
    return [row, column]

with open('mazes/testmaze.txt', 'r') as maze_file:
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

def determine_path_to_corrupt():
    '''determine_path_to_corrupt() -> Path
    outputs path in MDP that is best for adversary to corrupt'''
    P_p = P_star
    for P_i in paths: # don't iterate over best path
        if P_i.reward == P_star.reward: continue
        b_i = 2
        for P_j in paths:
            if P_j.reward == P_i.reward or P_j.reward == P_star.reward: continue

            opt = 0 # just need to initialize to something, is really infinity though
            for state in P_j:
                if (state in P_i) != (state in P_star) and (traversal_factors[state] > opt):
                    deg_star = traversal_factors[state]
            
            b_i += traversal_factors[state]

        if P_i.reward < P_p.reward and P_star.reward - b_i*p*delta < P_i.reward:
            P_p = P_i

    return P_p

def preload_path_data(r, c, visited, current_path, factor):
    '''preload_path_data(int, int, set, arr, int) -> None
    recursively finds every path in MDP and stores each one's constituent states and total reward sum
    also calculates and stores traversal factor of every node'''
    traversal_factors[hash(r,c)] = factor
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
                new_factor = factor * 1/float(len(get_action_space((r,c))))
                preload_path_data(new_r, new_c, visited, deepcopy(current_path), new_factor)

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

def init():
    '''init() -> None
    general variable initiation and preprocessing protocol; sets up transition function and stores all paths in MDP'''
    global initial_transition_function, transition_function, P_star, path_to_corrupt
    initial_transition_function = create_initial_transition()
    transition_function = initial_transition_function

    preload_path_data(*start, set(), Path([], 0), 1)
    paths.sort(reverse=True) # best is first
    P_star = paths[0]
    path_to_corrupt = determine_path_to_corrupt()

def learn(num_episodes, gamma, epsilon, alpha, attack=False, maxent=False, victim=GREED, verbose=False, graph=False, lw=5, num_epochs=1):
    '''learn(int, float, float, float, bool) -> None
    trains global variable "values" to learn Q-values of maze'''
    global visited, values, transition_function

    label = "Greedy Victim, " if victim == GREED else "Random Victim, "
    label += "Adversary Present" if attack else "No Adversary Present"

    if verbose:
        print("COMMENCING TRAINING PROTOCOL\nNumber of Epochs: " + str(num_epochs) + "\nNumber of Episodes per Epoch: " + str(num_episodes))

    values = np.zeros((height, width, 4)) # intialize
    performance_history = np.zeros((num_epochs, num_episodes))
    for epoch in range(num_epochs):
        for episode in range(num_episodes):
            state = start
            if verbose:
                if episode % 100 == 0: print(episode)

            while not hash(*state) in terminals: # so long as we don't hit an end
                if victim == GREED:
                    action = best_action(state, epsilon) # get best action according to current policy

                else:
                    action = random.choice(get_action_space(state))
                reward, new_state = take_action(state, action) # take action and observe reward, new state

                if np.random.random() < p and attack:
                    if hash(*new_state) in path_to_corrupt and hash(*new_state) not in P_star:
                        reward += delta

                    elif hash(*new_state) not in path_to_corrupt and hash(*new_state) in P_star:
                        reward -= delta

                if maxent: reward = reward + entropy(get_policy(values, epsilon)[state[0]][state[1]]) # maxent transformation
                values[state[0]][state[1]][actions.index(action)] = \
                    (1-alpha) * get_value(state, action) + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
                state = new_state

            if graph:
                performance_history[epoch][episode] = evaluate()

    if graph:
        plt.plot(range(num_episodes), np.average(performance_history, axis=0), label=label, alpha=0.5, lw=lw)

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
    init()
    learn(100, 0.99, 0.3, 0.3, graph=True, attack=False, victim=RAND, lw=2, num_epochs=30)
    learn(100, 0.99, 0.3, 0.3, graph=True, attack=True, victim=GREED, lw=3, num_epochs=30)
    learn(100, 0.99, 0.3, 0.3, graph=True, attack=False, victim=GREED, lw=4, num_epochs=30)
    learn(100, 0.99, 0.3, 0.3, graph=True, attack=True, victim=RAND, lw=5, num_epochs=30)

    plt.legend(loc="upper right")
    plt.xlabel('Episode')
    plt.ylabel('Performance (Final Reward)')
    plt.show()

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