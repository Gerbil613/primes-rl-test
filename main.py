import numpy as np
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from path import Path

blocks = [] # list of invalid states (walls)
terminals = [] # list of terminal states
start = None # state at which we start
scores = {} # dict maps the hash of a state to the reward associated with it
transition_function = None # just declare

reward_deviation = 2

p = 0.5
delta = 5

paths = [] # stores array of Path objects
P_star = None
path_to_corrupt = None

height, width = 0, 0
actions = [[1,0],[0,1],[-1,0],[0,-1]] # action space (usually is subset of this b/c walls)
values = None
initial_values = None
traversal_factors = {}

def hash(a, b): # used as a hash function for states (which are represented by two independent numbers)
    return a * width + b

def unhash(hash):
    column = hash % width
    row = hash // width
    return [row, column]

def load_maze(string):
    '''load_maze(string) -> None
    reads in maze and initializes many different important global variables'''
    global grid, height, width, values, start, scores, terminals, blocks
    grid = []
    with open(string, 'r') as maze_file:
        for line in maze_file.readlines():
            line = line.strip('\n').split(' ')
            grid.append(line)

    height, width = len(grid), len(grid[0])
    values = np.zeros((height, width, 4)) # will be initialized as np.array of shape (height, width), outputs value

    row = 0
    for line in grid:
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
        row += 1

def determine_path_to_corrupt():
    '''determine_path_to_corrupt() -> Path
    outputs path in MDP that is best for adversary to corrupt'''
    P_p = P_star
    for P_i in paths: # don't iterate over best path
        if P_i.reward == P_star.reward: continue
        b_i = traversal_factors[P_star.states[-1]] + traversal_factors[P_i.states[-1]] # sum traversal factors of final state in each path
        for P_j in paths:
            if P_j.reward == P_i.reward or P_j.reward == P_star.reward: continue

            opt = 0 # just need to initialize to something, is really infinity though
            for state in P_j:
                if (state in P_i) != (state in P_star) and traversal_factors[state] > opt and state != hash(*start):
                    opt = traversal_factors[state]
            
            b_i += opt

        if P_i.reward < P_p.reward and P_star.reward - b_i*p*delta < P_i.reward:
            P_p = P_i

    return P_p

def preload_path_data(r, c, visited, current_path, factor):
    '''preload_path_data(int, int, set, arr, int) -> None
    a) recursively finds every path in MDP and stores each one's constituent states and total reward sum
    b) calculates and stores traversal factor of every node
    c) sets up transition function'''
    global paths, transition_function

    traversal_factors[hash(r,c)] = factor
    if hash(r,c) in terminals: # reached end of path
        paths.append(current_path) # store the data
        return

    visited.add(hash(r,c))

    current_path.states.append(hash(r,c)) # add current state to path
    current_path.reward += scores[hash(r, c)] # add reward to running total

    new_factor = 0
    for action in actions:
        new_r, new_c =  r + action[0], c + action[1]
        if hash(new_r,new_c) not in blocks and hash(new_r,new_c) not in visited and new_r >= 0 and new_c >= 0 and new_r < height and new_c < width:
            new_factor += 1

    new_factor = factor * 1/float(new_factor)

    for action in actions: # try each action and see which next states we can go to according to TF
        new_r, new_c =  r + action[0], c + action[1]
        if hash(new_r,new_c) not in blocks and hash(new_r,new_c) not in visited and new_r >= 0 and new_c >= 0 and new_r < height and new_c < width:
            transition_function[hash(r,c)][actions.index(action)][hash(new_r,new_c)] = 1
            preload_path_data(new_r, new_c, visited, deepcopy(current_path), new_factor)

def init():
    '''init() -> None
    general variable initiation and preprocessing protocol; sets up transition function, reads maze data, and stores all paths in MDP'''
    load_maze('mazes/testmaze.txt')
    global transition_function, P_star, path_to_corrupt
    transition_function = np.zeros((width*height,len(actions), width*height))

    preload_path_data(*start, set(), Path([], 0), 1)
    paths.sort(reverse=True) # best is first
    P_star = paths[0]
    path_to_corrupt = determine_path_to_corrupt()

def estimate_rewards(num_warm_episodes, attack):
    '''estimate_rewards(int, bool) -> arr
    estimates the rewards of the mdp during warm start'''
    estimated_rewards = np.zeros((height, width, 4))
    times_visited = np.zeros((height, width))
    for warm_episode in range(num_warm_episodes):
        state = start
        while not hash(*state) in terminals:
            action = random.choice(get_action_space(state))
            reward, new_state = take_action(state, action)

            if np.random.random() < p and attack:
                if hash(*new_state) in path_to_corrupt and hash(*new_state) not in P_star:
                    reward += delta

                elif hash(*new_state) not in path_to_corrupt and hash(*new_state) in P_star:
                    reward -= delta

            times_visited[new_state[0]][new_state[1]] += 1
            n = float(times_visited[new_state[0]][new_state[1]])
            estimated_rewards[state[0]][state[1]][actions.index(action)] = reward / n + estimated_rewards[state[0]][state[1]][actions.index(action)] * (n-1)/n

            state = new_state

    return estimated_rewards

def initialize_q_values(estimated_rewards, current_reward, r, c, visited):
    '''initialize_q_values(arr, int, int, int, arr) -> arr
    recursive function that, given the rewards estimated during warm start, calculates q-values based on that'''
    global initial_values
    visited.add(hash(r,c))
    if hash(r,c) in terminals:
        return [current_reward]

    possibilities = []
    for action in actions:
        new_r, new_c = r + action[0], c + action[1]
        new_reward = current_reward + estimated_rewards[r][c][actions.index(action)]
        if hash(new_r, new_c) not in visited and hash(new_r, new_c) not in blocks and new_r >= 0 and new_c >= 0 and new_c < width and new_r < height:
            x = initialize_q_values(estimated_rewards, new_reward, new_r, new_c, visited)
            initial_values[r][c][actions.index(action)] = max(x) - current_reward
            possibilities.extend(x)

    return possibilities

def learn(num_episodes, gamma, epsilon, alpha, num_warm_episodes=50, attack=False, verbose=0, graph=False, lw=5, num_epochs=1):
    '''learn(int, float, float, float, bool) -> None
    trains global variable "values" to learn Q-values of maze'''
    global values, transition_function, initial_values

    label = "Greedy Victim, " if epsilon < 1 else "Random Victim, "
    label += "Adversary Present" if attack else "No Adversary Present"

    if verbose > 0:
        print("COMMENCING TRAINING PROTOCOL\nNumber of Epochs: " + str(num_epochs) + "\nNumber of Episodes per Epoch: " + str(num_episodes))

    performance_history = np.zeros((num_epochs, num_episodes))

    initial_values = np.zeros((height, width, 4))
    initialize_q_values(estimate_rewards(num_warm_episodes, attack), 0, *start, set()) # initialize via warm start

    for epoch in range(num_epochs):
        values = deepcopy(initial_values)
        for episode in range(num_episodes):
            state = start
            if verbose == 2:
                if episode % (num_episodes // 10) == 0:
                    print('Epoch: ' + str(epoch) + ' Episode: ' + str(episode))

            while not hash(*state) in terminals: # so long as we don't hit an end

                action = best_action(state, epsilon) # warm state for num_warm_episodes samples
                reward, new_state = take_action(state, action) # take action and observe reward, new state

                if np.random.random() < p and attack:
                    if hash(*new_state) in path_to_corrupt and hash(*new_state) not in P_star:
                        reward += delta

                    elif hash(*new_state) not in path_to_corrupt and hash(*new_state) in P_star:
                        reward -= delta

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
    performance = 0
    state = start
    # simply loop and keep on using policy to progress through maze
    while not hash(*state) in terminals:
        reward, new_state = take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    #print('Ended evaluation at: ' + str(state))
    return performance

def main():
    init()
    print(path_to_corrupt)
    learn(200, 0.99, 1, 0.3, graph=True, attack=False, lw=2, num_epochs=200, verbose=1)
    learn(200, 0.99, 0.3, 0.3, graph=True, attack=True, lw=3, num_epochs=200, verbose=1)
    learn(200, 0.99, 0.3, 0.3, graph=True, attack=False, lw=4, num_epochs=200, verbose=1)
    learn(200, 0.99, 1, 0.3, graph=True, attack=True, lw=5, num_epochs=200, verbose=1)

    plt.legend(loc="lower right")
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
    '''sample_transition_function(arr, arr) -> arr
    given state and action, finds new state based on transition function's probability distribution'''
    distribution = transition_function[hash(*state)][actions.index(action)]
    return unhash(np.random.choice(range(width*height), p=distribution))
  
def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    return np.random.normal(loc=scores[hash(new_state[0],new_state[1])], scale=reward_deviation)

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