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
with open('simplest_maze.txt', 'r') as maze_file:
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

values = np.zeros((height, width, 4)) # will be initialized as np.array of shape (height, width), outputs value
num_episodes = 10000 # number of training episodes
gamma = 0.99 # discount factor
epsilon = 0.7 # greed factor
alpha = 0.4 # learning rate

transition_function = None # just declare
'''current state,new state,action'''

def create_initial_transition():
    '''initializes the transition function (default setting is original deterministic)'''
    valid_state = True
    output = np.zeros(shape=(width*height,width*height,4))

    for l in range(len(blocks)):
        '''print(' blocks for loop ')'''
        for m in range(width*height):
            for n in range(4):
                output[rc_to_transition(unhash(blocks[l])[0],unhash(blocks[l])[1])][m][n]=-1
    for l in range(len(terminals)):
        '''print(' terminals for loop ')'''
        for m in range(width*height):
            for n in range(4):
                output[rc_to_transition(unhash(terminals[l])[0],unhash(terminals[l])[1])][m][n]=-1

    for i in range(width*height): 
        '''print(' 1st for loop ')'''
        for j in range(width*height):
            '''print(' 2nd for loop ')'''
            for k in range(4):
                '''print(' 3rd for loop ')'''              
                '''checking if the current element's current state is a wall or terminal state'''
                '''if it's a legit state, we set the corresponding new state
                print((transition_to_rc(i)[0]+actions[k][0]),(transition_to_rc(i)[1]+actions[k][1]))'''
                if output[i][j][k]==0 and rc_to_transition((transition_to_rc(i)[0]+actions[k][0]),(transition_to_rc(i)[1]+actions[k][1]))==j:
                    output[i][j][k]=1
                    '''print('filling in 1s')'''
    '''print('and we r out!')'''
    for l in range(len(blocks)):
        '''print(' blocks for loop ')'''
        for m in range(width*height):
            for n in range(4):
                output[rc_to_transition(unhash(blocks[l])[0],unhash(blocks[l])[1])][m][n]=0
    for l in range(len(terminals)):
        '''print(' terminals for loop ')'''
        for m in range(width*height):
            for n in range(4):
                output[rc_to_transition(unhash(terminals[l])[0],unhash(terminals[l])[1])][m][n]=0
                '''print('setting terminals to 0')'''

    return output

def init_transition():
    global initial_transition_function, transition_function
    transition_function = deepcopy(initial_transition_function)

def transition_to_rc(transition_index):
    '''converts transition matrix index number to row and column'''
    column = transition_index%width
    row = int((transition_index-column)/width)
    return[row,column]

def rc_to_transition(row,column):
    '''converts row and column to transition matrix index number (of current state)'''
    transition=column+row*width
    return transition

def unhash(hash):
    column=hash%99999
    row=int((hash-column)/99999)
    return[row,column]

def learn(num_episodes, gamma, epsilon, alpha):
    '''learn(int, float, float, float) -> None
    trains global variable "values" to learn Q-values of maze'''
    global visited, values, transition_function

    values = np.zeros((height, width, 4)) # intialize
    init_transition()
    for episode in range(num_episodes):
        state = start
        visited = set()
        while not is_terminal(state): # so long as we don't hit an end
            visited.add(hash(*state)) # mark that we visited here
            action = best_action(state, epsilon) # get best action according to current policy
            reward, new_state = take_action(state, action) # take action and observe reward, new state
            values[state[0]][state[1]][actions.index(action)] = \
                 (1-alpha) * get_value(state, action) + alpha * (reward + gamma * get_value(new_state, best_action(new_state, 0))) # fundamental bellman equation update
            set_transition_zero(state)
            state = new_state

        if episode%100==0: print(episode)

def set_transition_zero(state):
    '''setting already visited states to have transition probability 0'''
    transition_function[:width*height][rc_to_transition(state[0],state[1])][:4]=0    

def evaluate():
    '''evaluate() -> int
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    global visited, reward_deviation, trans_attack_prob
    performance = 0
    state = start
    init_transition()
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
    global initial_transition_function, transition_function
    initial_transition_function = create_initial_transition()

    data = []
    num_trials = 100
    for num_episodes in range(0, 71, 5):
        sum = 0
        for trial in range(num_trials):
            print(num_episodes, trial)
            learn(num_episodes, 0.99, 1, 0.4)
            sum += evaluate()
        data.append([num_episodes, sum / float(num_trials)])
    print(data)
    
def print_rewards(row, column, reward):
    '''print_rewards(int, int, int) -> None
    prints out all the rewards for every possible path in the maze'''
    if is_terminal([row, column]):
        print(reward)
        return

    visited.add(hash(row, column))
    for action in actions:
        if row + action[0] < 0 or row + action[0] >= height or column + action[1] < 0 or column + action[1] >= width: continue
        index = hash(row + action[0], column + action[1])
        if index not in blocks and index not in visited:
            print_rewards(row + action[0], column + action[1], reward + scores[index])

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
    new_state = sampleTransitionFunction(action, state)
    if len(get_action_space(state)) > 1 and random.random() < trans_attack_prob: # randomly choose
        action = random.choice(get_action_space(state))
        new_state = [state[0] + action[0], state[1] + action[1]]

    reward = get_reward(new_state)

    return [reward, new_state]

def sampleTransitionFunction(action, state):
    '''very sus sampling function to transition to next state according to the transition matrix probabilities and current state+action'''
    '''basically we loop thru the matrix for the column w/ fixed current state and action and add up the probabilities and see if our chosen random number falls under this range'''
    random_var = random.random()
    counter=0
    i=0
    action_number=0
    state_number=0
    new_state_transition=0
    action_number,state_number=getStateActionNumbers(state,action)

    for i in range(width*height):
        counter=counter+transition_function[state_number][i][action_number]
        
        if random_var<counter:
            new_state_transition=i
            break

    new_state_rc=transition_to_rc(new_state_transition)
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

    state_number = rc_to_transition(state[0],state[1])
    return[action_number,state_number]
  
def get_reward(new_state):
    '''get_reward(tuple) -> int
    computes and returns reward for entering new_state'''
    return scores[hash(new_state[0],new_state[1])]

def get_value(state, action):
    '''CHANGE THIS TO BE GET EXPECTED VALUE -- AS ACCORDING TO NONDETERMINISTIC TRANSITION MATRIX PROBABILITES!!'''
    '''get_value(tuple, tuple) -> int
    gives the value of the next state, given the current state we are in and the action we're about to take'''
    
    if action == [0,0]: return 0 # this happens when we get the value at a terminal state; the computer tries the dummy action [0,0] at terminals
    return values[state[0]][state[1]][actions.index(action)]

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