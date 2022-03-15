import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

def cantor_pair(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return int(1/2 * (a + b) * (a + b + 1) + b)

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
                blocks.append(cantor_pair(row, column))
            elif item == '*': # terminal state
                terminals.append(cantor_pair(row, column))
                scores[cantor_pair(row, column)] = 0 # no reward for start
            elif item == 's': # start state
                start = [row, column]
                scores[cantor_pair(row, column)] = 0 # no reward for terminal state itself

            else: # regular state
                scores[cantor_pair(row, column)] = int(item)
            
            column += 1

        width = column
        row += 1

    height = row

actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)

values = [] # will be initialized as np.array of shape (height, width), outputs value
num_episodes = 10000 # number of training episodes
gamma = 0.99 # discount factor
epsilon = 0.1 # greed factor
alpha = 0.5 # learning rate

def learn():
    '''learn() -> None
    trains global variabel "values" to learn Q-values of maze'''
    global visited, values

    values = np.random.random((height, width)) # intialize
    for episode in range(num_episodes):
        state = start
        visited = set()
        while not is_terminal(state): # so long as we don't hit an end
            visited.add(cantor_pair(*state)) # mark that we visited here
            action = best_action(state, epsilon) # get best action according to current policy
            reward, new_state = take_action(state, action) # take action and observe reward, new state
            skippers = [] # this is a list of consecutive states in whom we only have one valid action
            # keep on moving forward until we hit an actual junction in the maze
            while len(get_action_space(new_state)) == 1:
                skippers.append(new_state)
                visited.add(cantor_pair(*new_state))
                add_reward, new_state = take_action(new_state, get_action_space(new_state)[0])
                # we will pretend that all reward accumulated in this deterministic trajectory came from entering the last state; this avoids diluting/discriminating against rewards on longer paths
                reward += add_reward 

            values[new_state] = (1-alpha) * values[new_state] + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
            new_value = values[new_state] # go back through all the deterministic states we sped through and give them Q-values
            for i in range(len(skippers) - 1, -1, -1):
                values[skippers[i]] = new_value
                new_value -= scores[cantor_pair(*skippers[i])] # the Q-value of a deterministic state is the Q-value of the following state minus its reward

            state = new_state # update state

def evaluate():
    '''evaluate() -> None
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    global visited
    performance = 0
    state = start
    visited = set()
    # simply loop and keep on using policy to progress through maze
    while not is_terminal(state):
        visited.add(cantor_pair(*state))
        reward, new_state = take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    print('Ended evaluation at: ' + str(state))
    return performance

def main():
    learn()
    plt.imshow(values) # visualize the value function
    plt.show()
    print('Evaluated score: ' + str(evaluate()))

def is_blocked(state):
    '''is_blocked(tuple) -> bool
    outputs whether the state is a wall'''
    return cantor_pair(*state) in blocks

def is_terminal(state):
    '''is_terminal(tuple) -> bool
    outputs whether the state is a terminal state'''
    return cantor_pair(*state) in terminals

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamis function'''
    new_state = (state[0] + action[0], state[1] + action[1])

    reward = get_reward(new_state)
    return [reward, new_state]

def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    return scores[cantor_pair(*new_state)]

def get_value(state, action):
    '''get_value(tuple, tuple) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    return values[state[0]+action[0]][state[1]+action[1]]

def get_action_space(state):
    ''''get_action_space(tuple) -> arr
    outputs list of all possible actions that may be taken in state'''
    output = []
    global visited
    for action in actions:
        new_state = [state[0] + action[0], state[1] + action[1]]
        if not is_blocked(new_state) and cantor_pair(*new_state) not in visited and new_state[0] >= 0 and new_state[0] < height and new_state[1] >= 0 and new_state[1] < width:
            output.append(action)

    return output

def best_action(state, epsilon):
    '''best_action(tuple, float) -> tuple
    outputs the best action in the current state according to current value function, but takes random action with probability epsilon'''
    poss_actions = get_action_space(state)
    if random.random() < epsilon: return random.choice(poss_actions)
    if len(poss_actions) == 0:
        return [0,1]

    best_action, best_value = poss_actions[0], get_value(state, poss_actions[0])
    for action in poss_actions:
        if get_value(state, action) > best_value:
            best_value = get_value(state, action)
            best_action = action

    return best_action

if __name__ == '__main__':
    main()