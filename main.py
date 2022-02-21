import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt

width, height = 10, 10 # size of gridworld
start = (0, 0) # starting square (row, column) is top left
goal = (height - 1, width - 1)
blocks = [(8,9),(2,1),(7,2),(6,3)]
values = np.random.rand(width, height) # value at each square
policy = [[random.choice([[1,0],[0,1],[-1,0],[0,-1]]) for c in range(width)] for r in range(height)] # initiate random policy (policies are deterministic)
num_episodes = 100
alpha = 0.4 # learning rate
epsilon = 0.1 # using epsilon-greedy algorithm

def main():
    global epsilon
    values[goal[0]][goal[1]] = 1
    for episode in range(num_episodes):
        print(episode)
        epsilon *= 0.95
        state = [start[0], start[1]]
        trajectory = [state] # keep track of trajectory
        total_reward = 0
        while state[0] != goal[0] or state[1] != goal[1]: # keep on going as long as we don't reach goal
            if np.random.rand(1) >= epsilon:
                best_action = [0, 1]
                for action in [[1,0],[0,-1],[-1,0]]:
                    try: # in case we try to go out of bounds
                        if get_value(state, action) > get_value(state, best_action):
                            best_action = action
                    except IndexError: continue

            else:
                best_action = random.choice([(0,1),(1,0),(-1,0),(0,-1)])

            reward, state = take_action(state, best_action)
            trajectory.append(deepcopy(state))
            total_reward += reward

        for i in range(len(trajectory) - 2, 0, -1): # value iteration
            state = trajectory[i]
            later_state = trajectory[i+1]
            values[state[0]][state[1]] += alpha * (values[later_state[0]][later_state[1]] - values[state[0]][state[1]])

    print(values)
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