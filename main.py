import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

def hash(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return a * 99999 + b

blocks = [] # list of invalid states (walls)
terminals = [] # list of terminal states
start = None # state at which we start
scores = {} # dict maps the hash of a state to the reward associated with it
width, height = None, None
visited = set() # global var listing all the states we have visited in the current trajectory; useful for stopping us from going backwards
with open('maze.txt', 'r') as maze_file:
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

actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)

values = [] # will be initialized as np.array of shape (height, width), outputs value
num_episodes = 10000 # number of training episodes
gamma = 0.99 # discount factor
epsilon = 0.1 # greed factor
alpha = 0.4 # learning rate

reward_deviation = 10

def learn():
    '''learn() -> None
    trains global variable "values" to learn Q-values of maze'''
    global visited, values, gamma

    values = np.zeros((height, width)) # intialize
    count = 0
    for episode in range(num_episodes):
        state = start
        visited = set()
        while not is_terminal(state): # so long as we don't hit an end
            visited.add(hash(*state)) # mark that we visited here
            action = best_action(state, epsilon) # get best action according to current policy
            reward, new_state = take_action(state, action) # take action and observe reward, new state
                
            values[new_state[0]][new_state[1]] = (1-alpha) * values[new_state[0]][new_state[1]] + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
            state = new_state
            if state[0] == 3 and state[1] == 22: count += 1

        if episode % 100 == 0: print(episode)
    print('percent',count / num_episodes)

def evaluate():
    '''evaluate() -> None
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    global visited, reward_deviation
    performance = 0
    reward_deviation = 0
    state = start
    visited = set()
    # simply loop and keep on using policy to progress through maze
    while not is_terminal(state):
        visited.add(hash(*state))
        reward, new_state = take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    #print('Ended evaluation at: ' + str(state))
    return performance

def main():
    # try ten values of lamda 0.1 - 1
    # evaluate lamda on 5 trials, take median result
    learn()
    plt.imshow(values) # visualize the value function
    plt.show()
    print('Evaluated score: ' + str(evaluate()))

def is_blocked(state):
    '''is_blocked(tuple) -> bool
    outputs whether the state is a wall'''
    return hash(*state) in blocks

def is_terminal(state):
    '''is_terminal(tuple) -> bool
    outputs whether the state is a terminal state'''
    return hash(*state) in terminals

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamis function'''
    global visited
    new_state = [state[0] + action[0], state[1] + action[1]]
    reward = get_reward(new_state)

    return [reward, new_state]

def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    score = np.random.normal(loc=scores[hash(*new_state)], scale=reward_deviation)
    return score

def get_value(state, action):
    '''get_value(tuple, tuple) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    new_state = [state[0] + action[0], state[1] + action[1]]

    return values[new_state[0]][new_state[1]]

def get_action_space(state):
    ''''get_action_space(tuple) -> arr
    outputs list of all possible actions that may be taken in state'''
    if is_terminal(state): return []
    output = []
    global visited
    for action in actions:
        new_state = [state[0] + action[0], state[1] + action[1]]
        if not is_blocked(new_state) and hash(*new_state) not in visited and new_state[0] >= 0 and new_state[0] < height and new_state[1] >= 0 and new_state[1] < width:
            output.append(action)

    return output

def best_action(state, epsilon):
    '''best_action(tuple, float) -> tuple
    outputs the best action in the current state according to current value function, but takes random action with probability epsilon'''
    poss_actions = get_action_space(state)
    if random.random() < epsilon: return random.choice(poss_actions)
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