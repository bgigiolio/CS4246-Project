from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import MDP
from lines import plotActions
from mdp import state_dict_to_P, save_result, is_valid_policy
from mdp_toolbox_custom import ValueIteration, PolicyIteration, QLearning
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from eval import Evaluator
from functional_approximation_solving import Environement

def main():
    ### CREATE DATASET ###
    # ### SEA ###
    latitude = (-12.5, 31.5)
    longitude = (88.5, 153)
    scale = .5
    goal = (95, -5.5)

        # ### DEMO ###
        scale = 1
        longitude = (86, 90)
        latitude = (-12, -8)
        goal = (88, -10)
        
        dataset=Dataset(longitude[0], longitude[1], latitude[0], latitude[1])
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=start, goal=goal)
        dataset.add_trafic_density(method="local_averege") 
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        #dataset.save("dataset_main")
    dataset=read_dataset("dataset_main")
    print(dataset.states[dataset.get_closest_state(90, 0)]['neighbours']) #this is working as intended

    if True:
        ###CREATE RISK TABLE ###

        # ### DEMO ###
        scale = 1
        longitude = (86, 90)
        latitude = (-12, -8)
        goal = (88, -10)
        start= (86, -12)

        a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=goal)
        print(a.stateToRisk(10)) ### USE THIS TO GET RISK AT A STATE
        print(a.indexToCoord)
        print(a.coordToIndex)

    if True:
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}
        P, R = state_dict_to_P(a.coordToIndex, dataset.states, actions, f"mdp_params/{longitude}_{latitude}_{scale}/")
        print(P, R)

        ### SOLVE MDP using MDP toolbox ###
        vi = ValueIteration(P, R, 0.95)
        vi.run()
        # print(vi.policy)
        # print(vi.V_avg)

    ### SAVE THE RESULTS 
    save_result(vi, f"results/{longitude}_{latitude}_{scale}_{vi.label}/")

    ### EVALUATE POLICY ###
    if True:
        evaluator = Evaluator(scale, epochs=5, epoch_duration=30)
        print("Path score: ", evaluator.evalPolicy(vi.policy, a.indexToCoord))



    visualize = False
    if visualize:
        ### VISUALIZE DATASET ###
        #plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)
        #plot_dataset_on_map(dataset, Attribute="density", Ranges=5) #- working as intended 


        ### VISUALIZE POLICY ###
        dataset_solved=read_dataset("functional_approximation_solved_demo")
        plot_dataset_on_map(dataset_solved, Attribute="action", Ranges=3)

        #the animation we discussed

main()