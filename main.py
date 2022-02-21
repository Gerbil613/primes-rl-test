import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

width, height = 5, 5 # size of gridworld
start = (0, 0) # starting square (row, column) is top left
goal = (height - 1, width - 1)
blocks = [(8,9),(2,1),(7,2),(6,3)]
values = np.random.rand(width, height) # value at each square
actions = [[1,0],[0,1],[-1,0],[0,-1]]
policy = [[random.choice(actions) for c in range(width)] for r in range(height)] # initiate random policy (policies are deterministic)
num_episodes = 100

def main():
    values[goal[0]][goal[1]] = 1
    for i in range(num_episodes): # policity iteration
        for row in range(height): # evaluation
            for column in range(width):
                state = [row, column]
                reward = 1 if state[0] == goal[0] and state[1] == goal[1] else -1
                values[row][column] = reward + take_action(state, policy[row][column])[0] # evaluation

        for row in range(height): # improvement
            for column in range(width):
                state = [row, column]
                best_action = [-1,0] if row > 0 else [1,0]
                for action in actions:
                    try:
                        if get_value(state, action) > get_value(state, best_action):
                            best_action = action
                    except IndexError: pass

                policy[row][column] = best_action

    print(np.array(policy))
    plt.imshow(values)
    plt.show()

def take_action(state, action):
    '''take_action(arr, arr) -> int, arr
    inputs current state and action to take, and outputs new state and reward acquired in the process
    this is transitions dynamis function'''
    new_state = (state[0] + action[0], state[1] + action[1])
    if new_state[0] >= height or new_state[0] < 0 or new_state[1] >= width or new_state[1] < 0 or new_state in blocks:
        new_state = state

    reward = 1 if new_state[0] == goal[0] and new_state[1] == goal[1] else -1
    return reward, new_state

def get_value(state, action):
    '''get_value(arr, arr) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    return values[state[0] + action[0]][state[1] + action[1]]

if __name__ == '__main__':
    main()