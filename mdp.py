#import mdptoolbox.mdp as mdp
import mdp_toolbox_custom as mdp
from pprint import pprint
import numpy as np
import json
import os

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
def state_dict_to_P(coord_to_index_map: dict, states: dict, actions: dict, folder_path: str = None):
    A_count = len(actions)
    P = np.zeros((A_count, len(states)))
    R = np.zeros((len(states), A_count))

    # print(states)
    # print()
    # print(coord_to_index_map)

    for coord in states:
        curr_state = coord_to_index_map[coord[0]][coord[1]]
        neighbour_lst = states[coord]['neighbours']
        action_set = set(range(A_count))
        for neighbour in neighbour_lst:
            action = neighbour[1]
            neighbour_state = coord_to_index_map[neighbour[0][0]][neighbour[0][1]]
            P[action][curr_state] = neighbour_state
            action_set.remove(action)
            R[curr_state][action] = 0 - states[neighbour[0]]['to_goal']

        for remaining_action in action_set:
            P[remaining_action][curr_state] = curr_state # set invalid actions to result back to current state
            R[curr_state][remaining_action] = 0 - states[coord]['to_goal']
            #print(curr_state, remaining_action)
    
    if (not folder_path):
        try:  
            os.mkdir(folder_path)  
        except OSError as error: 
            print(error) 

        with open(folder_path + "JSON.json", "w+") as f:
            json.dump({"P": P.tolist(), "R": R.tolist()}, f)

    return P, R

def main():
    grid = np.array([[0.1,0.2],[0.3,0.4]])
    actions = [((0,-1), [[0,1,0],[0,0,0],[0,0,0]]), ((1,0), [[0,0,0],[0,0,1],[0,0,0]])]
    P,R = generate_P_R_matrix(grid, actions)

    P = np.array([[2, 3, 0, 1], [1, 0, 3, 2]])
    R = np.array([[0.3, 0.2], [0.4, 0.1], [0.1, 0.4], [0.2, 0.3]])

    print(P)
    vi = mdp.FiniteHorizon(P, R, 0.95, 5)
    vi.run()
    print(vi.policy)

if __name__=='__main__':
    main()