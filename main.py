from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import generateFrame
from mdp import state_dict_to_P
from mdp_toolbox_custom import ValueIteration, PolicyIteration

def main():
    ### CREATE DATASET ###
    if True:
        dataset=Dataset(88.6, 152.9, -12.4, 31.3) #South East Asia
        scale = 1
        longitude = (88.6, 90)
        lattitude = (-12.4, -8.4)
        #dataset=Dataset(88.6, 90, -12.4, -8.4)
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        dataset.save("dataset_1")
    dataset=read_dataset("dataset_1")
    print(dataset.states[dataset.get_closest_state(90, 0)]['neighbours']) #this is working as intended

    if False:
        ###CREATE RISK TABLE ###
        lattitude=(dataset.min_lat, dataset.max_lat) #TODO, move this into the function
        scale=dataset.scale
        longitude=(dataset.min_lon, dataset.max_lon)
        generateFrame(lattitude, longitude, scale, data=dataset).to_csv(f"riskMaps/{lattitude}_{longitude}_{scale}.csv")

    if False:
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}
        P, R, index_to_coord_map = state_dict_to_P(dataset.states, actions)
        print(P, R)

        ### save MDP ### #TODO
        #not the gamma

        ### LOAD MDP ### #TODO

        ### SOLVE MDP using MDP toolbox ###
        vi = PolicyIteration(P, R, 0.95)
        vi.run()
        print(vi.policy)

    ### SAVE THE RESULTS #TODO
    # Pontus think at least policy, utilities for every step, steps_to_convergence

    visualize=True
    if visualize:
        ### VISUALIZE DATASET ###
        plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)

        ### VISUALIZE POLICY ###
        #the animation we discussed

main()