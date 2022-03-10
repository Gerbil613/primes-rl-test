import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

def cantor_pair(a, b):
    return int(1/2 * (a + b) * (a + b + 1) + b)

blocks = []
terminals = []
start = None
scores = {}
width, height = None, None
visited = set()
with open('maze.txt', 'r') as maze_file:
    row = 0
    for line in maze_file.readlines():
        line = line.strip('\n').split(' ')
        column = 0
        for item in line:
            if item == '|':
                blocks.append(cantor_pair(row, column))
            elif item == '*':
                terminals.append(cantor_pair(row, column))
                scores[cantor_pair(row, column)] = 0 # no reward for start
            elif item == 's':
                start = [row, column]
                scores[cantor_pair(row, column)] = 0 # no reward for terminal state itself

            else:
                scores[cantor_pair(row, column)] = int(item)
            
            column += 1

        width = column
        row += 1

    height = row

actions = [[1,0],[0,1],[-1,0],[0,-1]]

# - robustness? what attacks work in either settings
# - how should adversary structure their attack?
# - not convoluted adversary, mess with just reward
# - random rewards?
# - compare to optimal paths

# why isn't the RL consistently able to reach the optimal path?
# is this an effective model for rl?

values = []
num_episodes = 1000
gamma = 0.99
epsilon = 0.1
alpha = 0.1

def learn():
    global visited, values

    values = np.random.random((height,width)) # value at each square
    for episode in range(num_episodes):
        state = start
        visited = set()
        while not is_terminal(state):
            visited.add(cantor_pair(*state))
            reward, new_state = take_action(state, best_action(state, epsilon))
            values[new_state[0]][new_state[1]] = (1-alpha) * values[new_state[0]][new_state[1]] + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0)))
            state = new_state

        print(episode)

def evaluate():
    global visited
    performance = 0
    state = start
    visited = set()
    while not is_terminal(state):
        visited.add(cantor_pair(*state))
        reward, new_state = take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    return performance

def main():
    learn()
    print(evaluate())
    plt.imshow(values)
    plt.show()

def is_blocked(state):
    return cantor_pair(*state) in blocks

def is_terminal(state):
    return cantor_pair(*state) in terminals

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamis function'''
    new_state = (state[0] + action[0], state[1] + action[1])

    reward = get_reward(new_state)
    return [reward, new_state]

def get_reward(new_state):
    return scores[cantor_pair(*new_state)]

def get_value(state, action):
    '''get_value(arr, arr) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    global values
    new = [state[0] + action[0], state[1] + action[1]]
    if new in blocks:
        raise ValueError('blocks are not valid states')

    return values[new[0]][new[1]]

def get_action_space(state):
    output = []
    global visited
    for action in actions:
        new_state = [state[0] + action[0], state[1] + action[1]]
        if not is_blocked(new_state) and cantor_pair(*new_state) not in visited and new_state[0] >= 0 and new_state[0] < height and new_state[1] >= 0 and new_state[1] < width:
            output.append(action)

    return output

def best_action(state, epsilon):
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