from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import MDP
from lines import plotActions
from mdp import state_dict_to_P, save_result, is_valid_policy
from mdp_toolbox_custom import ValueIteration, PolicyIteration, QLearning
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from eval import Evaluator

def main():
    ### CREATE DATASET ###
    # ### SEA ###
    latitude = (-12.5, 31.5)
    longitude = (88.5, 153)
    scale = .5
    start = (89, -12)
    goal = (150, 20)

    # ### DEMO ###
    # scale = 1
    # longitude = (86, 90)
    # latitude = (-12, -8)
    # start = (86, -12)
    # goal = (88, -10)

    if False:
        dataset=Dataset(longitude[0], longitude[1], latitude[0], latitude[1])
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        dataset.save("dataset_1")

    else:
        dataset=read_dataset("dataset_1")

    print(dataset.states[dataset.get_closest_state(86, -10)]['neighbours']) #this is working as intended

    if True:
        ###CREATE RISK TABLE ###
        a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=goal)
        print(a.stateToRisk(10)) ### USE THIS TO GET RISK AT A STATE

    if True:
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}
        P, R = state_dict_to_P(a.coordToIndex, a.stateToRisk, dataset.states, actions, f"mdp_params/{longitude}_{latitude}_{scale}/")
        #print(P, R)

        ### save MDP ### #TODO
        #not the gamma
        

        ### LOAD MDP ### #TODO

        ### SOLVE MDP using MDP toolbox ###
        vi = ValueIteration(P, R, 0.95)
        vi.run()
        # print(vi.policy)
        # print(vi.V_avg)

    invalid_coords = is_valid_policy(vi.policy, a.indexToCoord, dataset.states, R)

    ### SAVE THE RESULTS #TODO
    # Pontus think at least policy, utilities for every step, steps_to_convergence
    save_result(vi, f"results/{longitude}_{latitude}_{scale}_{vi.label}/")

    ### EVALUATE POLICY ###
    if True:
        evaluator = Evaluator(scale, epochs=5, epoch_duration=30)
        print("Path score: ", evaluator.evalPolicy(vi.policy, a.indexToCoord))

    visualize = False
    if visualize:
        ### VISUALIZE DATASET ###
        plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)

        ### VISUALIZE POLICY ###
        #the animation we discussed

    if True:
        map = Basemap(llcrnrlon=longitude[0], llcrnrlat=latitude[0], urcrnrlon=longitude[1], urcrnrlat=latitude[1])
        map.drawcoastlines()
        plotActions(map, start=start, end=goal, coords=a.indexToCoord, policyFunction=vi.policy, granularity=scale)
        plt.show()

main()