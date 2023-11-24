from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map, apply_dataset_on_map
from buildRiskTable import MDP
from lines import plotActions, mapUtility, mapDanger, maplgDanger
from mdp import *
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from eval import Evaluator
from functional_approximation_solving import Environement
from pathRunner import runPath, coordToPolicy
import numpy as np

def main():
    ### CREATE DATASET ###
    ### SEA ###  ##scale 0.5##
    # latitude = (-12.5, 31.5)
    # longitude = (88.5, 153)
    # scale = .5
    # goal = (104.5, 1.5)
    # start = (146.5, -9.5)

    # the case study in the presentation
    latitude = (-12.4, 31.4)
    longitude = (88.4, 153)
    scale = .2
    goal = (104.8, 1.4) #Singapore
    start = (146.8, -10.0) #New Guinea
    
    # #a smaller scale problem Sulawei to Singapore
    # latitude=(-8,10)
    # longitude=(100,130)
    # scale = .5
    # goal = (105, 1.4) #approximatley Singapore
    # start=(1.6, 124.5) #Manado, an interesting starting place because Borneo is in the way so you can take two ways around 

        # ### DEMO ###
    # scale = .5
    # longitude = (90, 110)
    # latitude = (-12, 10)
    # start = (99.5, 4)
    # goal = (96, -5)

    DIR_NAME = f"{longitude}_{latitude}_{scale}_{goal}" #everything with density added from now on

    if True:
        # dataset=Dataset(longitude[0], longitude[1], latitude[0], latitude[1]) #here ranges are used!
        # dataset.generate_states(distance=scale) #needs to be done first
        # dataset.load_pirate_data(spread_of_danger=1)
        # dataset.set_start_goal_generate_distance(start=start, goal=goal)

        dataset=read_dataset(f"{longitude}_{latitude}_{scale}_{goal}")
        # dataset.add_trafic_density(method="local_averege") ###THIS IS EXTREMLY FAST
        #print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        # dataset.save(DIR_NAME)
    else:
        dataset=read_dataset(DIR_NAME)


    #print(dataset.states[(100, -1)])

    # return

    if True:
        ###CREATE RISK TABLE ###

        a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=None, folder_path=DIR_NAME) #EVERY TIME NEW DATA this MUST BE DONE
        #print(a.stateToRisk(10)) ### USE THIS TO GET RISK AT A STATE
        #print(a.indexToCoord)
        #print(a.coordToIndex)
    
    else:
        a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=None, folder_path=DIR_NAME, read_file=True)

        goal_state = a.coordToIndex[goal[0]][goal[1]]
        print(goal_state)
    
    # return

    if False: 
        #MDP pipeline
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}

        """
        P is A x S matrix, where P[a][s] is state obtained from performing action a in state s
        R(s,a) is reward obtained from performing action a from state s
        """
        P, R = state_dict_to_P(a.coordToIndex, a.stateToRisk, dataset.states, actions, goal_state, DIR_NAME)
        print(P, R)
    else:
        P, R = read_mdp_params(DIR_NAME)
        # print(P, R)
    
    # return

    if False:
        ### SOLVE MDP using MDP toolbox ###
        ## label values: VI, PI, QL, SARSA, DQN
        label = "VI"
        ## VALUE ITERATION
        match label:
            case "VI":
                V, policy = value_iteration(P, R, epsilon=1e-8)
                label = "VI"
            case "PI":
                V, policy = policy_iteration(P, R)
                label = "PI"
            case "QL":
                V, policy = Q_learning(P, R, terminal_state=goal_state, 
                                       num_episodes=20000, reduction_factor=1, 
                                       epsilon_greedy=0.3, alpha=0.4,
                                       timeout = 10)
                label = "QL"
            case "SARSA":
                V, policy = SARSA(P, R, terminal_state=goal_state, num_episodes=5000)
                label = "SARSA"
            case "DQN":
                V, policy = DQN(P, R, terminal_state=goal_state)
                label = "DQN"

        save_result(policy, V, label, DIR_NAME)
    else:
        label = "QL"
        label = "QL"
        V, policy = read_result(label, DIR_NAME)

    #policy_adj = fix_policy(policy, start, goal, a.coordToIndex, a.indexToCoord, dataset.states)
    #print(np.array_equal(policy, policy_new))

    # return

    if True:
        ### EVALUATE POLICY ###
        coordToPolicy(a.indexToCoord, policy)
        path = runPath(policy=policy, start=start, goal=goal, coordToIndex=a.coordToIndex, scale=scale)
        evaluator = Evaluator(scale, epochs=1, epoch_duration=10223)
        print(f"Path Score: {evaluator.evalPolicy(path)}")
        # evaluator.largeAverage(100, path)
        # coordToPolicy(a.indexToCoord, policy)

    if False: #functional approximation example
        environment=Environement(DIR_NAME)
        environment.encode(environment.dataset.goal, one_hot_encoding=True) #initialization
        environment.set_model()
        environment.train()
        policy, utility=environment.generate_policy_utility()
        policy_ind=coordToPolicy(a.indexToCoord, policy)
        utility_ind=coordToPolicy(a.indexToCoord, utility) #should work

        ### Plotting the utility of the functional approximation policy and the path ###
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])
        map.drawcoastlines()
        plotActions(map, start=start, end=goal, coords=a.indexToCoord, policyFunction=policy_ind, granularity=scale) #a line from start to goal
        mapUtility(map, value_policy=utility_ind,index_to_coords=a.indexToCoord, size=100) #size here is the scaling of the scatter points
        #map.plot([goal[0], start[0]], [goal[1], start[1]], color="g", latlon=True) #shortest path between start and stop
        plt.show()

    if True:
        ### VISUALIZE DATASET ###
        plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)
        plot_dataset_on_map(dataset, Attribute="density", Ranges=5) #- working as intended 


        ### VISUALIZE POLICY ###
        dataset_solved=read_dataset("functional_demo", "functional_approximation_solved_demo")
        plot_dataset_on_map(dataset_solved, Attribute="action", Ranges=3, size=10) 

        #the animation we discussed
    
    if True:
        ### Plot Line ###
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=dataset.max_lon, urcrnrlat=dataset.max_lat) #instead of longitude[1], latitude[1], but it was not the issue
        map.drawcoastlines()
        map.drawmapboundary(fill_color='aqua')
        map.fillcontinents(color='lightgreen')
        plotActions(map, start=start, end=goal, coords=a.indexToCoord, policyFunction=policy, granularity=scale)
        # map.plot([goal[0], start[0]], [goal[1], start[1]], color="g", latlon=True) #shortest path between start and stop
        # map.scatter(goal[0] - 0.1, goal[1] + 0.7, marker=11, s=100, color="r")
        # plt.annotate("Singapore", (goal[0] - 2.6, goal[1] + 1))
        plt.show()

    if False:
        ### Display utility ###
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])
        map.drawcoastlines()
        mapUtility(map, value_policy=V,index_to_coords=a.indexToCoord, size=9)
        plt.show()

    if False:
        ### Display line on danger map###
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])
        map.drawcoastlines()
        mapDanger(map, dataset=dataset, size=9)
        plotActions(map, start=start, end=goal, coords=a.indexToCoord, policyFunction=policy, granularity=scale)
        plt.show()

    if False:
        ### Display utility and line ###
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])
        map.drawcoastlines()
        mapUtility(map, value_policy=V,index_to_coords=a.indexToCoord, size=1)
        plotActions(map, start=start, end=goal, coords=a.indexToCoord, policyFunction=policy, granularity=scale)
        plt.show()


main()