import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

blocks = []
goals = []
start = None
bad_goals = []
width, height = None, None
with open('maze.txt', 'r') as maze_file:
    row = 0
    for line in maze_file.readlines():
        line = line.strip('\n').split(' ')
        column = 0
        for item in line:
            if item == '1':
                blocks.append([row, column])
            elif item == '*':
                goals.append([row, column])
            elif item == 'x':
                bad_goals.append([row, column])
            elif item == 's':
                start = [row, column]
            column += 1

        width = column
        row += 1

    height = row

values = np.random.random((height,width)) # value at each square
actions = [[1,0],[0,1],[-1,0],[0,-1]]
policy = None # initialize
# - robustness? what attacks work in either settings
# - how should adversary structure their attack?
# - not convoluted adversary, mess with just reward
# examine the way rewards are rewarded (why always -1)
# - random rewards?
# - compare to optimal paths
num_episodes = 1000
gamma = 0.99
epsilon = 0

def main():
    policy = [[random.choice(actions) if not is_blocked(r,c) else None for c in range(width)] for r in range(height)] # initiate random policy (policies are deterministic)
    for i in range(num_episodes): # policy iteration
        for row in range(height): # evaluation
            for column in range(width):
                if is_blocked(row, column): continue
                state = [row, column]
                reward = 1 if state in goals else (-1 if state in bad_goals else 0)
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