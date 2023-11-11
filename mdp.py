#import mdptoolbox.mdp as mdp
import mdp_toolbox_custom as mdp
from pprint import pprint
import numpy as np
import json
import os
import tqdm
import math
import random
from collections import deque 
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from time import time

"""
P is NOT probability matrix, it is transition matrix
P[i][j] is the result state of taking aciton i from state j
"""  
def state_dict_to_P(coord_to_index_map: dict, index_to_reward_func: callable, states: dict, actions: dict, goal_state, folder_path: str = None):
    A_count = len(actions)
    P = np.zeros((A_count, len(states)))
    R = np.zeros((len(states), A_count))

    goal_reward = 1000
    penalty_for_moving = -5
    penalty_for_not_moving = -10000

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

            #R[curr_state][action] = float(index_to_reward_func(neighbour_state))
            #R[curr_state][action] = -states[neighbour[0]]['to_goal']
            if (neighbour_state == goal_state):
                R[curr_state][action] = goal_reward + penalty_for_moving
            else:
                R[curr_state][action] = penalty_for_moving - float(index_to_reward_func(neighbour_state))

        for remaining_action in action_set:
            P[remaining_action][curr_state] = curr_state # set invalid actions to result back to current state
            R[curr_state][remaining_action] = penalty_for_not_moving
            #R[curr_state][remaining_action] = states[coord]['to_goal']
            #print(curr_state, remaining_action)

    
    if (folder_path):
        try:  
            os.makedirs(f"mdp_params/{folder_path}", exist_ok=True)  
        except OSError as error: 
            print(error) 

        with open(f"mdp_params/{folder_path}/JSON.json", "w+") as f:
            json.dump({"P": P.tolist(), "R": R.tolist()}, f)

    return P.astype(int), R.astype(float)

def read_mdp_params(folder_path):
    with open(f'mdp_params/{folder_path}/JSON.json') as f:
        d = json.load(f)
        P = np.asarray(d["P"], dtype=np.int64)
        R = np.asarray(d["R"], dtype=np.float64)

    return P, R

CONST_GAMMA = 0.95
CONST_EPSILON = 1e-16
CONST_THETA = 1e-16
CONST_MAX_ITER = 10000
CONST_ALPHA = 0.01
CONST_EPSILON_GREEDY = 0.01
CONST_EPISODES = 1000
CONST_TIEMOUT = 60

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
    Q_values_lst = []
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values = [1 * (rewards[s][a] + gamma * V[transitions[a][s]]) for a in range(num_actions)]
        policy[s] = np.argmax(Q_values)
        Q_values_lst.append(list(Q_values))
    
    return V, policy, Q_values_lst

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

def Q_learning(transitions, rewards, terminal_state = 0, gamma = CONST_GAMMA, 
               alpha = CONST_ALPHA, epsilon = CONST_EPSILON, epsilon_greedy = CONST_EPSILON_GREEDY, 
               num_episodes = CONST_EPISODES, max_iter = CONST_MAX_ITER, reduction_factor = 1.0001,
               exploration_param = 1, timeout = CONST_TIEMOUT):
    num_actions = len(transitions)
    num_states = len(rewards)
    max_iter = num_states * num_states
    action_counts = np.zeros((num_states, num_actions))

    Q = np.zeros((num_states, num_actions))

    goal_state_counter = 0
    timeout_counter = 0
    max_iter_counter = 0

    for _ in tqdm.tqdm(range(num_episodes)):
        delta = 0
        s = np.random.randint(0, num_states)
        visited_states = {}

        timeout_thres = time() + timeout
        for i in range(max_iter):
            if np.random.uniform(0, 1) < epsilon_greedy:
                action = np.random.randint(0, num_actions)  # Random action
            else:
                # action = np.argmax(Q[s])
                exploration_values = Q[s] + exploration_param * np.sqrt(np.log(np.sum(action_counts[s])) / (1 + action_counts[s]))
                action = np.argmax(exploration_values)

            # if  (transitions[action][s] in visited_states):
            #     flag = False
            #     arr = np.arange(num_actions)
            #     np.random.shuffle(arr)
            #     for a in arr:
            #         action = a
            #         if  (not transitions[action][s] in visited_states):
            #             flag = True
            #             break
            #     if (not flag):
            #         #print(f"invalid action encountered {s}, continuing")
            #         break

            next_state = transitions[action][s]
            reward = rewards[s][action]
            visited_states[next_state] = True

            max_next_action = np.max(Q[next_state])

            Q[s][action] += alpha * (reward + gamma * max_next_action - Q[s][action])
            action_counts[s][action] += 1
            delta = max(delta, np.abs(alpha * (reward + gamma * max_next_action - Q[s][action])))

            if (np.isnan(Q[s][action])):
                print("problem")
                print(alpha * (reward + gamma * max_next_action))
                break

            #print(Q[s])
            s = next_state

            if next_state == terminal_state:  # terminal_state is the state where the episode ends
                goal_state_counter += 1
                break

            if (time() >= timeout_thres):
                #print(f"timeout encountered, breaking iter {i}...")
                timeout_counter += 1
                break

            if (i == max_iter - 1):
                max_iter_counter += 1

            # if (time() >= timeout - 55):
            #     print(f"timeout encountered, {i} {s}...")
            #     print(Q[s])
            #     print(np.transpose(transitions)[s])              
    
        alpha /= reduction_factor
        epsilon_greedy /= reduction_factor

        #print()
        # if (delta < epsilon):
        #     break

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        V[s] = np.max(Q[s])
        policy[s] = np.argmax(Q[s])

    print(f"Goal State Counter: {goal_state_counter}")
    print(f"Timeout Counter: {timeout_counter}")
    print(f"Max Iter Counter: {max_iter_counter}")
    return V, policy

def SARSA(transitions, rewards, terminal_state = 0, gamma = CONST_GAMMA, alpha = CONST_ALPHA, epsilon = CONST_EPSILON, epsilon_greedy = CONST_EPSILON_GREEDY, num_episodes = CONST_EPISODES, max_iter = CONST_MAX_ITER):
    num_actions = len(transitions)
    num_states = len(rewards)

    Q = np.zeros((num_states, num_actions))

    for episode in tqdm.tqdm(range(num_episodes)):
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
        # if (delta < epsilon):
        #     break

    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        V[s] = np.max(Q[s])
        policy[s] = np.argmax(Q[s])

    return V, policy

class DQNAgent:
    def __init__(self, state_size, action_size, transition_matrix, reward_func, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.transition_matrix = transition_matrix
        self.reward_func = reward_func
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', input_shape=[1]))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print(state, action, reward, next_state, done)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose = 0)[0])
            target_f = self.model.predict(np.array([state]), verbose = 0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose = 0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def simulate_environment(self, state, action, terminal_state):
        # Simulate environment using transition matrix and reward function
        next_state = self.transition_matrix[action][state]
        reward = self.reward_func[state][action]
        done = next_state == terminal_state

        return next_state, reward, done

def get_optimal_policy(agent, num_states):
    optimal_policy = []
    for state in range(num_states):
        q_values = agent.model.predict(np.reshape(state, [1, num_states]))[0]
        action = np.argmax(q_values)
        optimal_policy.append(action)

    return optimal_policy

def DQN(transitions, rewards, terminal_state = 0, gamma = CONST_GAMMA):
    num_actions = len(transitions)
    num_states = len(rewards)

    agent = DQNAgent(num_states, num_actions, transitions, rewards, gamma)

# Training loop
    EPISODES = 1000  # Number of episodes
    BATCH_SIZE = 32

    for episode in tqdm.tqdm(range(EPISODES), "Episode"):
        state = 0  # Assuming initial state is 0
        done = False
        total_reward = 0
        
        while not done:
            # Take an action using epsilon-greedy policy
            action = agent.act(state)
            
            # Simulate the environment based on the action
            next_state, reward, done = agent.simulate_environment(state, action, terminal_state)
            
            # Store the experience in the agent's memory
            agent.remember(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if len(agent.memory) > BATCH_SIZE:
                # Train the agent by replaying experiences
                agent.replay(BATCH_SIZE)
                print(agent.memory)
        
        # Decay exploration rate
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
    #return get_optimal_policy(agent, num_states)

def save_result(policy: np.ndarray, V: np.ndarray, label:str, folder_path: str):
    try:
        os.makedirs(f"results/{folder_path}_{label}", exist_ok=True)  
    except OSError as error: 
        print(error)

    with open(f"results/{folder_path}_{label}/JSON.json", "w+") as f:
        json.dump({"policy": policy.tolist(), "utility": V.tolist(), "iter": len(V)}, f)

def read_result(label, folder_path):
    with open(f'results/{folder_path}_{label}/JSON.json') as f:
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

def fix_policy(policy, start, goal, coord_to_index_map, index_to_coord_map, states, Q_values_lst = None):
    visited_states = {}
    start_state = coord_to_index_map[start[0]][start[1]]
    goal_state = coord_to_index_map[goal[0]][goal[1]]
    curr_state = start_state
    policy_new = np.copy(policy)

    while curr_state != goal_state:
        proposed_action = policy_new[curr_state]
        curr_coord = index_to_coord_map[curr_state]
        neighbour_lst = [neighbour[0] for neighbour in states[curr_coord]['neighbours']]
        neighbour_action_lst = [neighbour[1] for neighbour in states[curr_coord]['neighbours']]
        dest_coord = neighbour_lst[neighbour_action_lst.index(proposed_action)]
        dest_state = coord_to_index_map[dest_coord[0]][dest_coord[1]]

        if (dest_state in visited_states):
            print(neighbour_lst, neighbour_action_lst, proposed_action)

            neighbour_action_lst.remove(proposed_action)
            neighbour_lst.remove(dest_coord)
            if (Q_values_lst != None):
                Q_values_lst[dest_state][proposed_action] = -math.inf
                new_action = np.argmax(Q_values_lst[dest_state])
                #print(Q_values_lst[dest_state])
            else:
                new_action = random.choice(neighbour_action_lst)

            dest_coord = neighbour_lst[neighbour_action_lst.index(new_action)]
            print(new_action, dest_coord)
            dest_state = coord_to_index_map[dest_coord[0]][dest_coord[1]]

            policy_new[curr_state] = new_action
        
        curr_state = dest_state
        visited_states[dest_state] = True

        print(index_to_coord_map[curr_state])
    
    return policy_new


# def main():
#     grid = np.array([[0.1,0.2],[0.3,0.4]])
#     actions = [((0,-1), [[0,1,0],[0,0,0],[0,0,0]]), ((1,0), [[0,0,0],[0,0,1],[0,0,0]])]
#     P,R = generate_P_R_matrix(grid, actions)

#     neg_reward = -1000

#     P = np.array([[2, 3, 2, 3], [1, 1, 3, 3]])
#     R = np.array([[0.1, 0.1], [10, neg_reward], [neg_reward, 10], [neg_reward, neg_reward]])

#     print(P)
#     print(R)
#     # vi = mdp.FiniteHorizon(P, R, 0.95, 5)
#     # vi.run()
#     # print(vi.policy)

#     V, policy = Q_learning(P, R, terminal_state=3)
#     print(V, policy)

# if __name__=='__main__':
#     main()