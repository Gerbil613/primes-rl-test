import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from path import Path
from mdp import MDP

mdp = MDP(string='mazes/testmaze.txt')
path_to_corrupt = None

initial_values = None
values = None

p = 1
delta = 5

def determine_path_to_corrupt():
    '''determine_path_to_corrupt() -> Path
    outputs path in MDP that is best for adversary to corrupt'''
    P_p = mdp.P_star
    for P_i in mdp.paths: # don't iterate over best path
        if P_i.reward == mdp.P_star.reward: continue
        b_i = mdp.traversal_factors[mdp.P_star.states[-1]] + mdp.traversal_factors[P_i.states[-1]] # sum traversal factors of final state in each path
        for P_j in mdp.paths:
            if P_j.reward == P_i.reward or P_j.reward == mdp.P_star.reward: continue

            opt = 0 # just need to initialize to something, is really infinity though
            for state in P_j:
                if (state in P_i) != (state in mdp.P_star) and mdp.traversal_factors[state] > opt and state != mdp.start:
                    opt = mdp.traversal_factors[state]
            
            b_i += opt

        if P_i.reward < P_p.reward and mdp.P_star.reward - b_i*p*delta < P_i.reward:
            P_p = P_i

    return P_p

def estimate_rewards(num_warm_episodes, attack):
    '''estimate_rewards(int, bool) -> arr
    estimates the rewards of the mdp during warm start'''
    estimated_rewards = np.zeros((len(mdp.states), 4))
    times_visited = np.zeros((len(mdp.states)))
    for warm_episode in range(num_warm_episodes):
        state = mdp.start
        while len(mdp.get_action_space(state)) > 0:
            action = np.random.choice(mdp.get_action_space(state))
            reward, new_state = mdp.take_action(state, action)

            if np.random.random() < p and attack:
                if new_state in path_to_corrupt and new_state not in mdp.P_star:
                    reward += delta

                elif new_state not in path_to_corrupt and new_state in mdp.P_star:
                    reward -= delta

            times_visited[new_state] += 1
            n = float(times_visited[new_state])
            estimated_rewards[state][action] = reward / n + estimated_rewards[state][action] * (n-1)/n

            state = new_state

    return estimated_rewards

def initialize_q_values(estimated_rewards, current_reward, state, visited):
    '''initialize_q_values(arr, int, int, int, arr) -> arr
    recursive function that, given the rewards estimated during warm start, calculates q-values based on that'''
    global initial_values
    visited.add(state)

    possibilities = []
    action_space = mdp.get_action_space(state)
    if len(action_space) == 0: return [current_reward]
    for action in action_space:
        new_state = mdp.sample_transition_function(state, action)
        new_reward = current_reward + estimated_rewards[state][action]
        if new_state not in visited and new_state in mdp.states:
            x = initialize_q_values(estimated_rewards, new_reward, new_state, visited)
            initial_values[state][action] = max(x) - current_reward
            possibilities.extend(x)

    return possibilities

def learn(num_episodes, gamma, epsilon, alpha, num_warm_episodes=50, attack=False, verbose=0, graph=False, lw=5, num_epochs=1):
    '''learn(int, float, float, float, bool) -> None
    trains global variable "values" to learn Q-values of maze'''
    global values, initial_values

    label = "Greedy Victim, " if epsilon < 1 else "Random Victim, "
    label += "Adversary Present" if attack else "No Adversary Present"

    if verbose > 0:
        print("COMMENCING TRAINING PROTOCOL\nNumber of Epochs: " + str(num_epochs) + "\nNumber of Episodes per Epoch: " + str(num_episodes))

    performance_history = np.zeros((num_epochs, num_episodes))

    initial_values = np.zeros((len(mdp.states), len(mdp.actions)))
    initialize_q_values(estimate_rewards(num_warm_episodes, attack), 0, mdp.start, set()) # initialize via warm start

    for epoch in range(num_epochs):
        values = deepcopy(initial_values)
        for episode in range(num_episodes):
            state = mdp.start
            if verbose == 2:
                if episode % (num_episodes // 10) == 0:
                    print('Epoch: ' + str(epoch) + ' Episode: ' + str(episode))

            while len(mdp.get_action_space(state)) > 0: # so long as we don't hit an end
                action = best_action(state, epsilon) # warm state for num_warm_episodes samples
                reward, new_state = mdp.take_action(state, action) # take action and observe reward, new state

                if np.random.random() < p and attack:
                    if new_state in path_to_corrupt and new_state not in mdp.P_star:
                        reward += delta

                    elif new_state not in path_to_corrupt and new_state in mdp.P_star:
                        reward -= delta

                if len(mdp.get_action_space(new_state)) > 0:
                    values[state][action] = \
                    (1-alpha) * values[state][action] + alpha * (reward + gamma * values[new_state][best_action(new_state, 0)]) # fundamental bellman equation update
                
                else: # q-value of next state is 0 if no actions can be taken in next_state
                    values[state][action] = \
                    (1-alpha) * values[state][action] + alpha * reward
                state = new_state

            if graph:
                performance_history[epoch][episode] = evaluate()

    if graph:
        plt.plot(range(num_episodes), np.average(performance_history, axis=0), label=label, alpha=0.5, lw=lw)

def evaluate():
    '''evaluate() -> int
    evaluates the global var "values" according to a deterministic (non-epsilon) greedy policy'''
    performance = 0
    state = mdp.start
    # simply loop and keep on using policy to progress through maze
    while len(mdp.get_action_space(state)) > 0:
        reward, new_state = mdp.take_action(state, best_action(state, 0))
        performance += reward
        state = new_state

    #print('Ended evaluation at: ' + str(state))
    return performance

def main():
    global path_to_corrupt
    path_to_corrupt = determine_path_to_corrupt()
    learn(200, 0.99, 1, 0.3, graph=True, attack=False, lw=2, num_epochs=200, verbose=1)
    learn(200, 0.99, 0.3, 0.3, graph=True, attack=True, lw=3, num_epochs=200, verbose=1)
    learn(200, 0.99, 0.3, 0.3, graph=True, attack=False, lw=4, num_epochs=200, verbose=1)
    learn(200, 0.99, 1, 0.3, graph=True, attack=True, lw=5, num_epochs=200, verbose=1)

    plt.legend(loc="lower right")
    plt.xlabel('Episode')
    plt.ylabel('Performance (Final Reward)')
    plt.show()

def best_action(state, epsilon):
    '''best_action(state, real number) -> tuple
    outputs the best action in the current state according to current value function, but takes random action with probability epsilon'''
    poss_actions = mdp.get_action_space(state)
   
    if np.random.random() < epsilon:
        return np.random.choice(poss_actions)

    if len(poss_actions) == 0:
        return [0,0]

    best_action, best_value = poss_actions[0], values[state][poss_actions[0]]
    for action in poss_actions:
        if values[state][action] > best_value:
            best_value = values[state][action]
            best_action = action
    return best_action

if __name__ == '__main__':
    main()