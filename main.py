import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

width, height = 10, 10 # size of gridworld
start = (0, 0) # starting square (row, column) is top left
goal = (height - 1, width - 1)
num_blocks = 5
blocks_coords = np.random.choice(range(1, width * height - 1), size=num_blocks, replace=False) # assumes start and goal are top-left and bottom right, respectively
blocks = [[[[r,c] for r in range(height)] for c in range(width)][j // height][j % height] for j in blocks_coords]
values = np.zeros((height,width)) # value at each square
actions = [[1,0],[0,1],[-1,0],[0,-1]]
policy = None # initialize
# epsilon greedy?
# - robustness? what attacks work in either settings
# - how should adversary structure their attack?
# - not convoluted adversary, mess with just reward
# mess with gamma
# non-equal length paths (non-rectangular): what happens with gamma
# - what does messing with gamma do?
# examine the way rewards are rewarded (why always -1)
# - random rewards?
# there are so many optimal paths
num_episodes = 1000
gamma = 1
epsilon = 0.1

def main():
    policy = [[random.choice(actions) if not is_blocked(r,c) else None for c in range(width)] for r in range(height)] # initiate random policy (policies are deterministic)
    values[goal[0]][goal[1]] = 1
    for i in range(num_episodes): # policy iteration
        for row in range(height): # evaluation
            for column in range(width):
                if is_blocked(row, column): continue
                state = [row, column]
                reward = 1 if state[0] == goal[0] and state[1] == goal[1] else -1
                new_action = policy[row][column] if np.random.rand() > epsilon else random.choice(actions)
                values[row][column] = reward + gamma * get_value(state, new_action) # evaluation

                best_action = [-1,0] if row > 0 else [1,0]
                for action in actions:
                    if get_value(state, action) > get_value(state, best_action):
                        best_action = deepcopy(action)

                policy[row][column] = best_action

    for row in range(height):
        for column in range(width):
            if is_blocked(row, column):
                print('-- ', end='')
            else:
                if policy[row][column] == [0,1]:
                    print('-> ',end='')
                
                elif policy[row][column] == [0,-1]:
                    print('<- ', end='')

                elif policy[row][column] == [-1,0]:
                    print('^^ ',end='')

                elif policy[row][column] == [1,0]:
                    print('⌄⌄ ',end='')
        print('\n')

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
    new = [state[0] + action[0], state[1] + action[1]]
    if is_blocked(*new):
        return values[state[0]][state[1]]

    if new[0] >= 0 and new[0] < height and new[1] >= 0 and new[1] < width:
        return values[state[0] + action[0]][state[1] + action[1]]

    return values[state[0]][state[1]]

def is_blocked(row, column):
    for block in blocks:
        if row == block[0] and column == block[1]:
            return True

    return False

if __name__ == '__main__':
    main()