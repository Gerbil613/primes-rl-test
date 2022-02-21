import numpy as np
from copy import deepcopy
import random

width, height = 5, 5
start = (0, 0)
goal = (height - 1, width - 1)
blocks = [(3,4),(2,1),(3,2)]
values = np.zeros((width, height))
num_episodes = 100
alpha = 0.1
epsilon = 0.1

def main():
    values[goal[0]][goal[1]] = 1
    for episode in range(num_episodes):
        state = [start[0], start[1]]
        trajectory = [state]
        total_reward = 0
        while state[0] != goal[0] or state[1] != goal[1]:
            if np.random.rand(1) >= epsilon:
                best_action = [0, 1]
                for action in [[1,0],[0,-1],[-1,0]]:
                    try:
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

def take_action(state, action):
    new_state = (state[0] + action[0], state[1] + action[1])
    if new_state[0] > 4 or new_state[0] < 0 or new_state[1] > 4 or new_state[1] < 0 or new_state in blocks:
        new_state = state

    reward = 1 if new_state[0] == goal[0] and new_state[1] == goal[1] else -1
    return reward, new_state

def get_value(state, action):
    return values[state[0] + action[0]][state[1] + action[1]]

if __name__ == '__main__':
    main()