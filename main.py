import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
from mdp import MDP
import seaborn as sns
import math
from tqdm import tqdm

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

def get_observed_path_rewards(test_corruption_algorithm):
    '''get_observed_path_rewards(arr) -> Path
    outputs list of Path objects whose rewards are augmented by coruption_algorithm'''
    global mdp
    observed_rewards = [0] * len(mdp.paths)
    for i in range(len(mdp.paths)):
        path = mdp.paths[i]
        for edge in path:
            observed_rewards[i] += mdp.rewards[edge[0], edge[1]] + p * np.average(test_corruption_algorithm[:, edge[0], edge[1]])

    return observed_rewards

def alicia_heuristic():
    '''alicia_heuristic() -> None
    implements alicia's heuristic version of algorithm 2 to deal with in-between paths'''
    global corruption_algorithm, path_to_corrupt, mdp
    corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states)))
    path_to_corrupt = mdp.P_star
    for P_p in mdp.paths: # P_p is tested for corruption
        test_corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states)))
        unique_edge = mdp.unique_edges[P_p.id]
        test_corruption_algorithm[P_p.id][unique_edge[0]][unique_edge[1]] = delta
        for interceding_path_index in range(len(mdp.paths) - 1, P_p.id, -1): # loop thru best to worst paths, trying to bring down each as conservatively as possible
            interceding_path = mdp.paths[interceding_path_index]

            observed_path_rewards = get_observed_path_rewards(test_corruption_algorithm)
            if observed_path_rewards[interceding_path.id] < observed_path_rewards[P_p.id]:
                break

            for free_corruption_path in mdp.paths: # loop thru all paths to get free corruption to bring down interceding path
                for edge in free_corruption_path:
                    if edge in P_p and edge not in interceding_path: # preference for corrupting up if possible
                        test_corruption_algorithm[free_corruption_path.id] = np.zeros((len(mdp.states), len(mdp.states))) # override any downward corruption from before
                        test_corruption_algorithm[free_corruption_path.id, edge] = delta

                    elif edge in interceding_path and edge not in P_p and not np.sum(test_corruption_algorithm[free_corruption_path.id]) != 0: # if already corrupted on this path, don't mess with it
                        test_corruption_algorithm[free_corruption_path.id][edge[0]][edge[1]] = -delta
                        break

                observed_path_rewards = get_observed_path_rewards(test_corruption_algorithm)
                if observed_path_rewards[interceding_path.id] < observed_path_rewards[P_p.id]:
                    break
                
        if np.argmax(get_observed_path_rewards(test_corruption_algorithm)) == P_p.id and path_to_corrupt.reward > P_p.reward:
            path_to_corrupt = deepcopy(P_p)
            corruption_algorithm = deepcopy(test_corruption_algorithm)

def algorithm2():
    '''algorithm2() -> None
    computes path in MDP that is best for adversary to corrupt and sets global variables
    Implements Algorithm 2'''
    global corruption_algorithm, path_to_corrupt, mdp
    P_p = mdp.P_star
    test_corruption_algorithm = np.zeros((len(mdp.paths), len(mdp.states), len(mdp.states)))
    for P_i in mdp.paths: # don't iterate over best path
        if P_i.id == mdp.P_star.id: continue

        result, budget = test_path(P_i)
        if P_i.reward < P_p.reward and mdp.P_star.reward - budget*p*delta < P_i.reward:
            test_corruption_algorithm = result
            P_p = P_i

    corruption_algorithm = deepcopy(test_corruption_algorithm)
    path_to_corrupt = deepcopy(P_p)

def test_path(test_path_to_corrupt):
    '''test_path(path) -> array, float
    determines which edge to corrupt for each path in the dynamic adversarial algorithm, in order to switch inputted path
    helper function for algorithm2'''
    test_corruption_algorithm = np.zeros((len(mdp.paths), int(mdp.transition_function.shape[0]), len(mdp.states))) # specify path and edge
    budget = 0
    for path in mdp.paths:
        opt = -1
        opt_edge = -1
        for edge in path:
            if (edge in test_path_to_corrupt) != (edge in mdp.P_star) and (opt == -1 or mdp.traversal_factors[edge[0]][edge[1]] < opt):
                opt = mdp.traversal_factors[edge[0]][edge[1]]
                opt_edge = edge

        budget += 1.0 / opt if opt != -1 else 0
        if opt_edge != -1: test_corruption_algorithm[path.id][opt_edge[0]][opt_edge[1]] = delta if opt_edge in test_path_to_corrupt else -delta
    
    return test_corruption_algorithm, budget

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
    num_mdps_per_step = 200
    num_reward_steps = 6
    mean_nodes_per_layer = 4
    num_steps = int(mean_nodes_per_layer / 2) + 1
    data = np.zeros((num_steps, num_reward_steps))
    num_layers = 3
    x, y = [], []
    for density_step in range(num_steps):
        mean_degree = mean_nodes_per_layer - num_steps + density_step + 1
        print('Mean degree:', mean_degree)
        y.append(str(mean_degree)[:4])
        numerator, denominator = 0, 0
        for reward_step in range(num_reward_steps):
            reward_std = 10**(reward_step/float(num_reward_steps) - 1/2.0)
            print('Reward deviation:',str(reward_std)[:5])
            if density_step == 0: x.append(str(reward_std)[:5])
            for i in tqdm(range(num_mdps_per_step)):
                mdp.load_random_layered(num_layers, mean_nodes_per_layer, mean_degree, reward_std*p*delta, assure_unique_edges=True)
                alicia_heuristic()
                observed_path_rewards = get_observed_path_rewards(corruption_algorithm)
                heuristic_result = mdp.paths[np.argmax(observed_path_rewards)].reward

                algorithm2()
                observed_path_rewards = get_observed_path_rewards(corruption_algorithm)
                alg2_result = mdp.paths[np.argmax(observed_path_rewards)].reward

                numerator += heuristic_result - alg2_result
                denominator += 1

            data[density_step][reward_step] = float(numerator) / denominator

    print('Finished computation.')
    sns.heatmap(data, annot=True)
    plt.xticks(range(len(x)), x)
    plt.yticks(range(len(y)), y)
    plt.title('Difference in reward of corrupted path between Heuristic and Algorithm 2')
    plt.xlabel('Reward deviation')
    plt.ylabel('Mean edge degree (density)')
    plt.show()

if __name__ == '__main__':
    main()