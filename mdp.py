#import mdptoolbox.mdp as mdp
import mdp_toolbox_custom as mdp
from pprint import pprint
import numpy as np
import json
import os
import tqdm
import math

"""
Convert basemap to xy grid, with - denoting obstacle and each cell is the reward
"""
def lat_lon_matrix_to_grid(basemap, nx, ny, metadata=None):
    lat_lon_grid = basemap.makegrid(nx, ny)
    len_x = len(lat_lon_grid[0][0])
    len_y = len(lat_lon_grid[0])

    xy_to_lat_lon_map = {}
    grid = [[0 for x in range(len_x)] for x in range(len_y)]
    for i in range(len_y):
        for j in range(len_x):
            xy_to_lat_lon_map[(j, i)] = (lat_lon_grid[0][i][j], lat_lon_grid[1][i][j])
            xpt, ypt = basemap(lat_lon_grid[0][i][j], lat_lon_grid[1][i][j])
            if (basemap.is_land(xpt, ypt)):
                grid[i][j] = '-'
            else:
                grid[i][j] = '0'
    
    return (grid, xy_to_lat_lon_map)

"""
:param _grid: denote obstacles with -, otherwise denote with reward
:param actions: array of (intended action in displacement tuple, 3x3 array starting at (1,1) probability of reaching each of the 8 surrounding cells)
:param reward_grid: xy array of rewards of each location
"""
def generate_P_R_matrix(grid, actions):    
    len_x = len(grid[0])
    len_y = len(grid)

    A_count = len(actions)

    state_to_coord_map = {}
    state_to_reward = {}
    coord_to_state_map = {}
    state_counter = 0

    for i in range(len_y):
        for j in range(len_x):
            if (grid[i][j] != "-"):
                state_to_coord_map[state_counter] = (j, i)
                coord_to_state_map[(j, i)] = state_counter
                state_to_reward[state_counter] = grid[i][j]
                state_counter += 1

    P = [[[0 for x in range(state_counter)] for y in range(state_counter)] for z in range(A_count)]
    R = [[0 for x in range(state_counter)] for y in range(A_count)]

    for a in range(A_count):
        for s in range(state_counter):
            action_grid = actions[a][1]
            current_state = s
            current_coord = state_to_coord_map[s]

            for i in range(len(action_grid)):
                for j in range(len(action_grid[0])):
                    d_x, d_y = j - 1, i - 1
                    target_x, target_y = d_x + current_coord[0], d_y + current_coord[1]

                    # circular array behaviour
                    target_x %= len_x
                    target_y %= len_y

                    target_state = current_state
                    if ((target_x, target_y) in coord_to_state_map):
                        target_state = coord_to_state_map[(target_x, target_y)]

                    P[a][current_state][target_state] += action_grid[i][j]
                    if (d_x == actions[a][0][0] and d_y == actions[a][0][1]):
                        R[a][current_state] = state_to_reward[target_state]
        
    return (np.asarray(P), np.transpose(np.asarray(R)))

"""
P is NOT probability matrix, it is transition matrix
P[i][j] is the result state of taking aciton i from state j
"""  
def state_dict_to_P(coord_to_index_map: dict, index_to_reward_func: callable, states: dict, actions: dict, folder_path: str = None):
    A_count = len(actions)
    P = np.zeros((A_count, len(states)))
    R = np.zeros((len(states), A_count))

    penalty_for_not_moving = -1e8

    # print(states)
    # print()
    # print(coord_to_index_map)

    for coord in tqdm.tqdm(states.keys(), "generating P and R matrix"):
        curr_state = coord_to_index_map[coord[0]][coord[1]]
        neighbour_lst = states[coord]['neighbours']
        action_set = set(range(A_count))
        
        for neighbour in neighbour_lst:
            action = neighbour[1]
            neighbour_state = coord_to_index_map[neighbour[0][0]][neighbour[0][1]]
            P[action][curr_state] = neighbour_state
            action_set.remove(action)
            try:
                #R[curr_state][action] = float(index_to_reward_func(neighbour_state))
                R[curr_state][action] = states[neighbour[0]]['to_goal']
            except:
                R[curr_state][action] = penalty_for_not_moving

        for remaining_action in action_set:
            P[remaining_action][curr_state] = curr_state # set invalid actions to result back to current state
            #R[curr_state][remaining_action] = penalty_for_not_moving
            R[curr_state][remaining_action] = states[coord]['to_goal']
            #print(curr_state, remaining_action)

    
    if (folder_path):
        try:  
            os.makedirs(folder_path, exist_ok=True)  
        except OSError as error: 
            print(error) 

        with open(folder_path + "JSON.json", "w+") as f:
            json.dump({"P": P.tolist(), "R": R.tolist()}, f)

    return P.astype(int), R.astype(float)

import numpy as np

CONST_GAMMA = 0.95
CONST_EPSILON = 1e-11
CONST_THETA = 1e-11
CONST_MAX_ITER = 10000
CONST_ALPHA = 0.05
CONST_EPSILON_GREEDY = 0.1
CONST_EPISODES = 100000

def value_iteration(transitions, rewards, gamma = CONST_GAMMA, epsilon = CONST_EPSILON, max_iter = CONST_MAX_ITER):
    num_actions = len(transitions)
    num_states = len(rewards)
    
    V = np.zeros(num_states)
    for i in tqdm.tqdm(range(max_iter), "iteration"):
        delta = 0
        for s in range(num_states):
            v = np.copy(V[s])
            Q_values = [1 * (rewards[s][a] + gamma * V[transitions[a][s]]) for a in range(num_actions)]
            V[s] = max(Q_values)
            delta = max(delta, abs(v - V[s]))

            #print(Q_values, V)
        
        if delta < epsilon:
            break
        #print()
    
    # Extract the optimal policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values = [1 * (rewards[s][a] + gamma * V[transitions[a][s]]) for a in range(num_actions)]
        policy[s] = np.argmax(Q_values)
    
    return V, policy

import numpy as np

def policy_evaluation(policy, transitions, rewards, gamma = CONST_GAMMA, theta = CONST_THETA):
    num_states = len(rewards)
    
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = np.copy(V[s])
            action = policy[s]
            
            next_state = transitions[action][s]  # Assuming a deterministic policy
            reward = rewards[s][action]
            V[s] = reward + gamma * V[next_state]
            
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
        
    return V

def policy_improvement(transitions, rewards, V, old_policy, gamma = CONST_GAMMA):
    num_actions = len(transitions)
    num_states = len(rewards)
    
    policy_stable = True
    new_policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        action_values = [1 * (rewards[s][a] + gamma * V[transitions[a][s]]) for a in range(num_actions)]
        new_policy[s] = np.argmax(action_values)
        
        if old_policy[s] != new_policy[s]:
            policy_stable = False
            
    return new_policy, policy_stable

def policy_iteration(transitions, rewards, gamma = CONST_GAMMA, max_iter = CONST_MAX_ITER):
    num_actions = len(transitions)
    num_states = len(rewards)
    
    policy = np.random.randint(num_actions, size=num_states)
    
    for i in tqdm.tqdm(range(max_iter), "iteration"):
        V = policy_evaluation(policy, transitions, rewards, gamma)
        new_policy, policy_stable = policy_improvement(transitions, rewards, V, policy, gamma)
        
        if policy_stable:
            break
        
        policy = new_policy
    
    return V, policy

def Q_learning(transitions, rewards, terminal_state = 0, gamma = CONST_GAMMA, alpha = CONST_ALPHA, epsilon = CONST_EPSILON, epsilon_greedy = CONST_EPSILON_GREEDY, num_episodes = CONST_EPISODES, max_iter = CONST_MAX_ITER):
    num_actions = len(transitions)
    num_states = len(rewards)

    Q = np.zeros((num_states, num_actions))

    for _ in tqdm.tqdm(range(num_episodes)):
        delta = 0
        s = np.random.randint(0, num_states)

        for i in range(max_iter):
            if np.random.uniform(0, 1) < epsilon_greedy:
                action = np.random.randint(0, num_actions)  # Random action
            else:
                action = np.argmax(Q[s])

            next_state = transitions[action][s]
            reward = rewards[s][action]

            max_next_action = np.max(Q[next_state])

            Q[s][action] += alpha * (reward + gamma * max_next_action - Q[s][action])
            delta = max(delta, np.abs(alpha * (reward + gamma * max_next_action - Q[s][action])))

            if (np.isnan(Q[s][action])):
                print("problem")
                print(alpha * (reward + gamma * max_next_action))
                break

            #print(Q[s])
            s = next_state

            if next_state == terminal_state:  # terminal_state is the state where the episode ends
                break
        #print()
        if (delta < epsilon):
            break

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        V[s] = np.max(Q[s])
        policy[s] = np.argmax(Q[s])

    return V, policy

def SARSA(transitions, rewards, terminal_state = 0, gamma = CONST_GAMMA, alpha = CONST_ALPHA, epsilon = CONST_EPSILON, epsilon_greedy = CONST_EPSILON_GREEDY, num_episodes = CONST_EPISODES, max_iter = CONST_MAX_ITER):
    num_states = len(transitions)
    num_actions = len(transitions[0])

    Q = np.zeros((num_states, num_actions))

    for episode in range(num_episodes):
        delta = 0
        s = np.random.randint(0, num_states) 

        if np.random.uniform(0, 1) < epsilon_greedy:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(Q[s])

        while True:
            next_state = transitions[action][s]
            reward = rewards[s][action]

            if np.random.uniform(0, 1) < epsilon_greedy:
                next_action = np.random.randint(0, num_actions)  # Random next action
            else:
                next_action = np.argmax(Q[next_state])

            Q[s][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[s][action])
            delta = max(delta, np.abs(alpha * (reward + gamma * Q[next_state][next_action] - Q[s][action])))

            s = next_state
            action = next_action

            if next_state == terminal_state:  # terminal_state is the state where the episode ends
                break
        if (delta < epsilon):
            break

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        V[s] = np.max(Q[s])
        policy[s] = np.argmax(Q[s])

    return V, policy

def save_result(policy: np.ndarray, V: np.ndarray, folder_path: str):
    try:  
        os.makedirs(folder_path, exist_ok=True)  
    except OSError as error: 
        print(error)

    with open(folder_path + "JSON.json", "w+") as f:
        json.dump({"policy": policy.tolist(), "utility": V.tolist(), "iter": len(V)}, f)

def read_result(folder_path):
    with open(f'{folder_path}/JSON.json') as f:
        d = json.load(f)
        policy = np.asarray(d["policy"], dtype=np.int64)
        V = np.asarray(d["utility"], dtype=np.float64)

    return V, policy

def is_valid_policy(policy, index_to_coord_map, states):
    invalid_coords = []

    for state in range(len(policy)):
        coord = (index_to_coord_map[state][0], index_to_coord_map[state][1])
        action = policy[state]
        list_of_valid_actions = [neighbour[1] for neighbour in states[coord]['neighbours']]

        if (action in list_of_valid_actions):
            pass
        else:
            invalid_coords.append(coord)

    return invalid_coords


def main():
    grid = np.array([[0.1,0.2],[0.3,0.4]])
    actions = [((0,-1), [[0,1,0],[0,0,0],[0,0,0]]), ((1,0), [[0,0,0],[0,0,1],[0,0,0]])]
    P,R = generate_P_R_matrix(grid, actions)

    neg_reward = -1000

    P = np.array([[2, 3, 2, 3], [1, 1, 3, 3]])
    R = np.array([[0.1, 0.1], [10, neg_reward], [neg_reward, 10], [neg_reward, neg_reward]])

    print(P)
    print(R)
    # vi = mdp.FiniteHorizon(P, R, 0.95, 5)
    # vi.run()
    # print(vi.policy)

    V, policy = Q_learning(P, R, terminal_state=3)
    print(V, policy)

if __name__=='__main__':
    main()