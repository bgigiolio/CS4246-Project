from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import MDP
from mdp import state_dict_to_P, save_result
from mdp_toolbox_custom import ValueIteration, PolicyIteration, QLearning

def main():
    ### CREATE DATASET ###
    if True:
        #dataset=Dataset(88.6, 152.9, -12.4, 31.3) #South East Asia
        scale = 1
        longitude = (86, 90)
        lattitude = (-12, -8)
        dataset=Dataset(86, 90, -12, -8)
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        #dataset.save("dataset_1")
    #dataset=read_dataset("dataset_1")
    print(dataset.states[dataset.get_closest_state(90, 0)]['neighbours']) #this is working as intended

    if True:
        ###CREATE RISK TABLE ###
        a = MDP(lat=lattitude, lon=longitude, scale=scale, data=dataset)
        #b = MDP(JSON_file="riskMaps\(-12.5, 20)_(88.5, 100)_0.5\JSON.json")
        

    if True:
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}
        P, R = state_dict_to_P(a.coordToIndex, dataset.states, actions, f"mdp_params/{longitude}_{lattitude}_{scale}/")
        print(P, R)

        ### save MDP ### #TODO
        #not the gamma
        

        ### LOAD MDP ### #TODO

        ### SOLVE MDP using MDP toolbox ###
        vi = ValueIteration(P, R, 0.95)
        vi.run()
        print(vi.policy)
        # print(vi.V_avg)

    ### SAVE THE RESULTS #TODO
    # Pontus think at least policy, utilities for every step, steps_to_convergence
    save_result(vi, f"results/{longitude}_{lattitude}_{scale}_{vi.label}/")

    visualize = False
    if visualize:
        ### VISUALIZE DATASET ###
        plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)

        ### VISUALIZE POLICY ###
        #the animation we discussed

main()