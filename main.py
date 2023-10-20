from dataset import Dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import generateFrame
from mdp import state_dict_to_P
from mdp_toolbox_custom import ValueIteration, PolicyIteration

def main():
    ### CREATE DATASET ###
    #dataset=Dataset(88.6, 152.9, -12.4, 31.3) #South East Asia
    scale = .1
    longitude = (88.6, 90)
    lattitude = (-12.4, -8.4)
    dataset=Dataset(88.6, 90, -12.4, -8.4)
    dataset.generate_states(distance=scale) #needs to be done first
    dataset.load_pirate_data(spread_of_danger=1)
    dataset.set_start_goal_generate_distance(start=(90, 0), goal=(150, 20))
    print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 

    ###CREATE RISK TABLE ###
    generateFrame(lattitude, longitude, scale, data=dataset).to_csv(f"riskMaps/{lattitude}_{longitude}_{scale}.csv")

    ### TRANSLATE DATASET TO MDP ###
    actions = {0: "right", 1: "up", 2: "left", 3: "down"}
    P, R, index_to_coord_map = state_dict_to_P(dataset.states, actions)
    print(P, R)

    ### SOLVE MDP using MDP toolbox ###
    vi = PolicyIteration(P, R, 0.95)
    vi.run()
    print(vi.policy)

    visualize=False
    if visualize:
        ### VISUALIZE DATASET ###
        plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)

        ### VISUALIZE POLICY ###

main()