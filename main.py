import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from path import Path
from mdp import MDP
import math
import pickle

mdp = MDP()
path_to_corrupt = None
corruption_algorithm = None

estimations = None

p = 1
delta = 2

def sample_truncanted_normal(mean, std, bound, precision=50):
    '''sample_truncated_normal(float, float, float) -> float
    samples from a normal distribution truncated between the minimum and maximum values
    bound must be positive'''
    assert bound >= 0
    x = np.arange(mean-bound, mean+bound, 2*bound/float(precision))
    y = 1 / std / (2*math.sqrt(math.pi)) * np.exp(-0.5 * ((x - mean) / std)**2)
    y /= np.sum(y)

    return np.random.choice(x, p=y)

def determine_path_to_corrupt():
    '''determine_path_to_corrupt() -> Path
    outputs path in MDP that is best for adversary to corrupt'''
    P_p = mdp.P_star
    for P_i in mdp.paths: # don't iterate over best path
        if P_i.reward == mdp.P_star.reward: continue
        b_i = 2
        for P_j in mdp.paths:
            if P_j.reward == P_i.reward or P_j.reward == mdp.P_star.reward: continue

            opt = 99999
            for edge in P_j:
                if (edge in P_i) != (edge in mdp.P_star) and mdp.traversal_factors[edge[0]][edge[1]] < opt:
                    opt = mdp.traversal_factors[edge[0]][edge[1]]
            
            b_i += 1/opt
        
        if P_i.reward < P_p.reward and mdp.P_star.reward - b_i*p*delta < P_i.reward:
            P_p = P_i

    return P_p

def determine_corruption_algorithm():
    '''determine_corruption_algorithm() -> dict
    determines which edge to corrupt for each path in the dynamic adversarial algorithm'''
    global corruption_algorithm
    corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states))) # specify path and edge
    for path in mdp.paths:
        for edge in path:
            if path.id == mdp.P_star.id:
                if mdp.traversal_factors[edge] == 1: # corrupt optimal path down
                    corruption_algorithm[path.id, edge[0], edge[1]] = -delta
                    break

            elif path.id == path_to_corrupt.id:
                if mdp.traversal_factors[edge] == 1: # corrupt path to switch up
                    corruption_algorithm[path.id, edge[0], edge[1]] = delta
                    break
            
            elif edge in mdp.P_star and edge not in path_to_corrupt: # free corruption
                    corruption_algorithm[path.id, edge[0], edge[1]] = -delta
                    break

            elif edge in path_to_corrupt and edge not in mdp.P_star: # free corruption
                corruption_algorithm[path.id, edge[0], edge[1]] = delta
                break

def learn(epsilon, num_warm_episodes=50, attack=0, verbose=1, graph=True, lw=5, num_epochs=500, num_episodes=200, objective_evaluation=True):
    '''learn(float, bool) -> None
    trains global variable "values" to learn Q-values of maze'''
    global estimations

    label = "Greedy Victim, " if epsilon < 1 else "Random Victim, "
    if attack == 0: label += "No Adversary, "
    elif attack == 1: label += "Attack A, "
    elif attack == 2: label += "Attack B, "
    elif attack == 3: label += "Dynamic Adversary, "
    label += "Warm Start, " if num_warm_episodes > 0 else "No Warm Start, "
    label += "Objective Evaluation" if objective_evaluation else "Victim's Perspective"

    if verbose > 0:
        print("COMMENCING TRAINING PROTOCOL\nNumber of Epochs: " + str(num_epochs) + "\nNumber of Episodes per Epoch: " + str(num_episodes))

    performance_history = np.zeros((num_epochs, num_episodes))

    for epoch in range(num_epochs):
        estimations = np.zeros((len(mdp.states), len(mdp.states)))
        times_visited = np.zeros((len(mdp.states), len(mdp.states)))

        for episode in range(num_episodes + num_warm_episodes):
            path = best_path(epsilon if episode >= num_warm_episodes else 1, estimations)
            can_attack = np.random.random() <= p
            differential = 0
            state = mdp.start
            for edge in path:
                state, new_state = edge[0], edge[1]
                times_visited[state, new_state] += 1
                reward = mdp.sample_reward_function(state, new_state)
                original_reward = reward

                if attack == 1 and can_attack:
                    reward += sample_truncanted_normal(0.0, 1.0, delta/float(len(path.states) - 1))
                
                if attack == 2 and can_attack:
                    if new_state in mdp.P_star:
                        reward -= delta/float(len(path.states) - 1) # subtract 1 to ensure delta bound; if a path contains n states, it contains n-1 edges

                    else:
                        reward += delta/float(len(path.states) - 1)

                if attack == 3 and can_attack:
                    reward += corruption_algorithm[path.id, edge[0], edge[1]]
                
                differential += abs(reward - original_reward)
                n = float(times_visited[state, new_state])
                estimations[state, new_state] = reward / n + estimations[state, new_state] * (n-1)/n
                state = new_state

            assert differential <= delta

            if graph:
                performance_history[epoch][episode - num_warm_episodes] = evaluate(estimations) if objective_evaluation else corrupted_evaluate(estimations)

    if graph:
        plt.plot(range(num_episodes), np.average(performance_history, axis=0), label=label, alpha=0.5, lw=lw)

def evaluate(estimations):
    '''evaluate() -> int
    evaluates reward estimations in unperturbed test-time setting'''
    return best_path(0, estimations).reward

def corrupted_evaluate(estimations):
    '''evaluate() -> int
    evaluates estimations from victim's corrupted perspective'''
    path = best_path(0, estimations)
    output = 0
    for state in path.states:
        output += estimations[state]

    return output

def best_path(epsilon, estimations):
    '''best_path(int, arr) -> path
    with probability epsilon, outputs random path in mdp
    with probability 1-epsilon, outputs path whose reward is best according to our estimations'''
    if np.random.random() < epsilon:
        return np.random.choice(mdp.paths)

    best_paths = [] # if multiple paths are estimated to have identical rewards and are also best, we choose randomly from that set
    best_reward = -9999
    for path in mdp.paths:
        estimated_reward = 0
        for edge in path:
            estimated_reward += estimations[edge]

        if estimated_reward > best_reward:
            best_reward = estimated_reward
            best_paths = [path]

        elif estimated_reward == best_reward:
            best_paths.append(path)
    
    return np.random.choice(best_paths)

def main():
    global path_to_corrupt
    #mdp.load_maze('mazes/testmaze.txt')
    np.random.seed(0)
    mdp.load_random(5)
    path_to_corrupt = determine_path_to_corrupt()
    determine_corruption_algorithm()
    print(corruption_algorithm)
    print('Number of paths:', len(mdp.paths))
    print('Path to switch:', path_to_corrupt)
    print('Optimal path:', mdp.P_star)
    print('Number of actions: ' + str(len(mdp.actions)))
    print('Score means:\n' + str(mdp.rewards))
    if input('Commence training? (y/n) ') == 'n': quit()
    attack = 3
    learn(0.3, 0, attack=attack, lw=1)
    learn(0.3, 50, attack=attack, lw=2)
    learn(1, 0, attack=attack, lw=3)
    learn(1, 50, attack=attack, lw=4)

    plt.legend(loc="lower right")
    plt.xlabel('Episode')
    plt.ylabel('Performance (Final Reward)')
    plt.show()

if __name__ == '__main__':
    main()