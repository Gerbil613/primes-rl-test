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
with open('super_simple_maze.txt', 'r') as maze_file:
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

reward_deviation = 0
trans_attack_prob = 0
transition_function = np.zeros(shape=(width*height,width*height,4))
'''current state,new state,action'''

def initTransition():
    '''initializes the transition function (default setting is original deterministic)'''
    valid_state = True
    print(' init Transition ')
    print(len(blocks))

    for l in range(len(blocks)):
        print(' blocks for loop ')
        for m in range(width*height):
            for n in range(4):
                transition_function[rcToTransition(unhash(blocks[l])[0],unhash(blocks[l])[1])][m][n]=-1
    for l in range(len(terminals)):
        print(' terminals for loop ')
        for m in range(width*height):
            for n in range(4):
                transition_function[rcToTransition(unhash(blocks[l])[0],unhash(blocks[l])[1])][m][n]=-1

    for i in range(width*height): 
        '''print(' 1st for loop ')'''
        for j in range(width*height):
            '''print(' 2nd for loop ')'''
            for k in range(4):
                '''print(' 3rd for loop ')'''              
                '''checking if the current element's current state is a wall or terminal state'''
                '''if it's a legit state, we set the corresponding new state
                print((transitionToRC(i)[0]+actions[k][0]),(transitionToRC(i)[1]+actions[k][1]))'''
                if transition_function[i][j][k]==0 and rcToTransition((transitionToRC(i)[0]+actions[k][0]),(transitionToRC(i)[1]+actions[k][1]))==j:
                    transition_function[i][j][k]=1
                    print('filling in 1s')
    print('and we r out!')



def transitionToRC(transition_index):
    '''converts transition matrix index number to row and column'''
    column = transition_index%width
    row = int((transition_index-column)/width)
    return[row,column]

def rcToTransition(row,column):
    '''converts row and column to transition matrix index number (of current state)'''
    transition=column+row*width
    return transition

def unhash(hash):
    column=hash%99999
    row=int((hash-column)/99999)
    return[row,column]

def learn():
    '''learn() -> None
    trains global variable "values" to learn Q-values of maze'''
    global visited, values, gamma

    values = np.zeros((height, width)) # intialize
    count = 0
    for episode in range(num_episodes):
        state = start
        visited = set()
        '''print(episode)'''
        while not is_terminal(state): # so long as we don't hit an end
            visited.add(hash(*state)) # mark that we visited here
            '''print('taking best action')'''
            action = best_action(state, epsilon) # get best action according to current policy
            '''print(action)'''
            reward, new_state = take_action(state, action) # take action and observe reward, new state
            '''print('reward')
            print(reward)'''
            values[new_state[0]][new_state[1]] = (1-alpha) * values[new_state[0]][new_state[1]] + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
            state = new_state
            if state[0] == 1 and state[1] == 7: count += 1

        if episode % 100 == 0: print(episode)
    print('percent',count / num_episodes)

def evaluate():
    '''evaluate() -> None
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    global visited, reward_deviation
    performance = 0
    reward_deviation = 0
    trans_attack_prob = 0
    state = start
    visited = set()
    # simply loop and keep on using policy to progress through maze
    while not is_terminal(state):
        visited.add(hash(*state))
        reward, new_state = take_action(state, best_action(state, 0))
        print(new_state)
        performance += reward
        state = new_state

    #print('Ended evaluation at: ' + str(state))
    return performance

def main():
    # try ten values of lamda 0.1 - 1
    # evaluate lamda on 5 trials, take median result
    print(' in the main ')
    initTransition()
    print('learning...')
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
    this is transitions dynamic function'''
    global visited
    '''new_state = [state[0] + action[0], state[1] + action[1]]'''
    new_state = sampleTransitionFunction(action, state)
    if len(get_action_space(state)) > 1 and random.random() < trans_attack_prob: # randomly choose
        action = random.choice(get_action_space(state))
        new_state = [state[0] + action[0], state[1] + action[1]]

    reward = get_reward(new_state)

    return [reward, new_state]

def sampleTransitionFunction(action, state):
    '''very sus sampling function to transition to next state according to the transition matrix probabilities and current state+action'''
    random_var = random.random()
    counter=0
    i=0
    action_number=0
    state_number=0
    new_state_transition=0
    action_number,state_number=getStateActionNumbers(state,action)
    '''    print('state,action numbers')
print(state_number)
    print(action_number)
    print(random_var)'''
    for i in range(width*height):
        counter=counter+transition_function[state_number][i][action_number]
        
        if random_var<counter:
            '''print('counter and i')
            print(counter)
            print(i)'''
            new_state_transition=i
            break

    new_state_rc=transitionToRC(new_state_transition)
    return new_state_rc

def getStateActionNumbers(state,action):
    action_number=0
    if action==actions[0]:
        action_number=0
    if action==actions[1]:
        action_number=1
    if action==actions[2]:
        action_number=2
    if action==actions[3]:
        action_number=3

    state_number = rcToTransition(state[0],state[1])
    return[action_number,state_number]

def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    score = np.random.normal(loc=scores[hash(*new_state)], scale=reward_deviation)
    return score

def get_value(state, action):
    '''CHANGE THIS TO BE GET EXPECTED VALUE -- AS ACCORDING TO NONDETERMINISTIC TRANSITION MATRIX PROBABILITES!!'''
    '''get_value(tuple, tuple) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    [action_number,state_number]=getStateActionNumbers(state,action)
    value=0
    for i in range (width*height):
        row,column=transitionToRC(i)
        value=value+transition_function[state_number][i][action_number]*values[row][column]
    return value
    '''print('value bruh')
    print(values[new_state[0]][new_state[1]])'''
'''
    return values[new_state[0]][new_state[1]]'''

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