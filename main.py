import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from path import Path
from mdp import MDP
import math
import time
import colorsys

mdp = MDP()
path_to_corrupt = None
corruption_algorithm = None

estimations = None

p = 1
delta = 1

def sample_truncanted_normal(mean, std, bound, precision=50):
    '''sample_truncated_normal(float, float, float) -> float
    samples from a normal distribution truncated between the minimum and maximum values
    bound must be positive'''
    assert bound >= 0
    x = np.arange(mean-bound, mean+bound, 2*bound/float(precision))
    y = 1 / std / (2*math.sqrt(math.pi)) * np.exp(-0.5 * ((x - mean) / std)**2)
    y /= np.sum(y)

    return np.random.choice(x, p=y)

def determine_path_to_corrupt(free_corruption=True):
    '''determine_path_to_corrupt() -> Path
    outputs path in MDP that is best for adversary to corrupt'''
    P_p = mdp.P_star
    test_corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states)))
    for P_i in mdp.paths: # don't iterate over best path
        if P_i.reward == mdp.P_star.reward: continue
        '''b_i = 2
        for P_j in mdp.paths:
            if P_j.reward == P_i.reward or P_j.reward == mdp.P_star.reward: continue

            opt = 99999
            for edge in P_j:
                if (edge in P_i) != (edge in mdp.P_star) and mdp.traversal_factors[edge[0]][edge[1]] < opt:
                    opt = mdp.traversal_factors[edge[0]][edge[1]]
            
            b_i += 1/opt'''

        result = determine_corruption_algorithm(P_i, free_corruption=free_corruption)
        if result[0] and P_i.reward < P_p.reward: # result[0] makes sure we surpass the in-between paths; is crucial
            test_corruption_algorithm = result[1]
            P_p = P_i

        '''if P_i.reward < P_p.reward and mdp.P_star.reward - b_i*p*delta < P_i.reward and P_i.reward + BLAH > mdp.paths[1].reward:
            P_p = P_i'''

    return P_p, deepcopy(test_corruption_algorithm)

def determine_corruption_algorithm(test_path_to_corrupt, free_corruption=True):
    '''determine_corruption_algorithm(path) -> np.array
    determines which edge to corrupt for each path in the dynamic adversarial algorithm'''
    test_corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states))) # specify path and edge
    budget_calculation = np.zeros((len(mdp.states), len(mdp.states)))
    for path in mdp.paths:
        for edge in path:
            if path.id == mdp.P_star.id:
                if mdp.traversal_factors[edge] == 1: # corrupt optimal path down
                    test_corruption_algorithm[path.id, edge[0], edge[1]] = -delta
                    budget_calculation[edge[0], edge[1]] = -p*delta
                    break

            elif path.id == test_path_to_corrupt.id:
                if mdp.traversal_factors[edge] == 1: # corrupt path to switch up
                    test_corruption_algorithm[path.id, edge[0], edge[1]] = delta
                    budget_calculation[edge[0], edge[1]] = p*delta
                    break
            
            elif edge in mdp.P_star and edge not in test_path_to_corrupt and free_corruption: # free corruption
                test_corruption_algorithm[path.id, edge[0], edge[1]] = -delta
                budget_calculation[edge[0], edge[1]] = -p*delta / mdp.traversal_factors[edge]
                break

            elif edge in test_path_to_corrupt and edge not in mdp.P_star and free_corruption: # free corruption
                test_corruption_algorithm[path.id, edge[0], edge[1]] = delta
                budget_calculation[edge[0], edge[1]] = p*delta / mdp.traversal_factors[edge]
                break
    
    path_increments = [np.sum([budget_calculation[edge] for edge in path]) for path in mdp.paths]
    corrupted_path_rewards = np.array([p.reward for p in mdp.paths]) + np.array(path_increments)
    return np.argmax(corrupted_path_rewards) == test_path_to_corrupt.id, test_corruption_algorithm

def learn(epsilon, num_warm_episodes=50, attack=0, verbose=1, lw=5, num_epochs=1000, num_greedy_episodes=200, objective_evaluation=True):
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
        print("COMMENCING TRAINING PROTOCOL\nNumber of Epochs: " + str(num_epochs) + "\nNumber of Episodes per Epoch: " + str(num_greedy_episodes + num_warm_episodes))
        print(label)

    output = np.zeros((num_epochs))

    for epoch in range(num_epochs):
        estimations = np.zeros((len(mdp.states), len(mdp.states)))
        times_visited = np.zeros((len(mdp.states), len(mdp.states)))

        for episode in range(num_greedy_episodes + num_warm_episodes):
            path = best_path(1 if episode < num_warm_episodes else epsilon, estimations)
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
                    if new_state in mdp.P_star.states:
                        reward -= delta/float(len(path.states) - 1) # subtract 1 to ensure delta bound; if a path contains n states, it contains n-1 edges

                    else:
                        reward += delta/float(len(path.states) - 1)

                if attack == 3 and can_attack:
                    reward += corruption_algorithm[path.id, edge[0], edge[1]]
                
                differential += abs(reward - original_reward)

                n = float(times_visited[state, new_state])
                estimations[state, new_state] = reward / n + estimations[state, new_state] * (n-1)/n
                state = new_state
            
            assert differential - delta <= 0.0001 # not quite 0 cuz floating point errors
        
        output[epoch] = evaluate(estimations)

    return output # return performance in every last episode

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
    # 3 independent variables - number of states, density, reward ratio
    global mdp, path_to_corrupt, corruption_algorithm
    num_steps = 20
    num_mdps_per_step = 2000
    x, y, z_four, z_five, z_six, z_seven = [], [], np.zeros((num_steps, num_steps)), np.zeros((num_steps, num_steps)), np.zeros((num_steps, num_steps)), np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        print(i)
        p_edge = 0.7*float(i) / num_steps + 0.3 # density
        x.append(p_edge)
        for j in range(num_steps):
            ratio = 10**(2*float(j) / (num_steps - 1) - 1)
            if i == 0: y.append(ratio)
            numerator, denominator = np.zeros((8)), np.zeros((8))
            for k in range(num_mdps_per_step):
                if k < num_mdps_per_step/4.0: num_states = 4
                elif k < num_mdps_per_step/2.0: num_states = 5
                elif k < 3*num_mdps_per_step/4.0: num_states = 6
                else: num_states = 7
                mdp.load_random(num_states, max_edge_reward=ratio*p*delta, p_edge=p_edge)
                path_to_corrupt_no_free, corruption_algorithm = determine_path_to_corrupt(free_corruption=0)
                path_to_corrupt_with_free, corruption_algorithm = determine_path_to_corrupt(free_corruption=1)
                if path_to_corrupt_with_free.id != mdp.P_star.id: denominator[num_states] += 1
                if path_to_corrupt_with_free.id != mdp.P_star.id and path_to_corrupt_with_free.id == path_to_corrupt_no_free.id: numerator[num_states] += 1

            z_four[j][i] = float(numerator[4]) / denominator[4]
            z_five[j][i] = float(numerator[5]) / denominator[5]
            z_six[j][i] = float(numerator[6]) / denominator[6]
            z_seven[j][i] = float(numerator[7]) / denominator[7]

    ax = plt.axes(projection='3d')
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, np.log10(y), z_four, cmap='Purples', linewidth=0, antialiased=False, label='4 States')
    ax.plot_surface(x, np.log10(y), z_five, cmap='Greens', linewidth=0, antialiased=False, label='5 States')
    ax.plot_surface(x, np.log10(y), z_six, cmap='Oranges', linewidth=0, antialiased=False, label='6 States')
    ax.plot_surface(x, np.log10(y), z_seven, cmap='Reds', linewidth=0, antialiased=False, label='7 States')
    ax.set_xlabel('Density')
    ax.set_ylabel('Ratio of max reward to pdelta')
    ax.set_zlabel("Percentage of time when allowing free corruption doesn't make a difference")
    plt.show()

if __name__ == '__main__':
    main()