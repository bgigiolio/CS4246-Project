from dataset import Dataset, read_dataset
from visualize_dataset import plot_dataset_on_map
from buildRiskTable import MDP
from lines import plotActions
from mdp import state_dict_to_P, save_result, read_result, is_valid_policy, value_iteration, policy_iteration, Q_learning, SARSA
from mdp_toolbox_custom import ValueIteration, PolicyIteration, QLearning
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from eval import Evaluator
from functional_approximation_solving import Environement
from pathRunner import runPath, coordToPolicy

def main():
    ### CREATE DATASET ###
    # ### SEA ###
    latitude = (-12.5, 31.5)
    longitude = (88.5, 153)
    scale = .5
    goal = (95, -5.5)
    start = (105, 0)

    if False:
        # ### DEMO ###
        # scale = 1
        # longitude = (86, 90)
        # latitude = (-12, -8)
        # goal = (88, -10)
        
        dataset=Dataset(longitude[0], longitude[1], latitude[0], latitude[1]) #here ranges are used!
        dataset.generate_states(distance=scale) #needs to be done first
        dataset.load_pirate_data(spread_of_danger=1)
        dataset.set_start_goal_generate_distance(start=start, goal=goal)
        # dataset.add_trafic_density(method="local_averege") 
        print(dataset) #this shows a random example state as well as all the parameters. Note that there is no indexing of the states at this part of the project. 
        dataset.save("dataset_1")
    else:
        dataset=read_dataset("dataset_1")

    if True:
        ###CREATE RISK TABLE ###

        a = MDP(lat=latitude, lon=longitude, scale=scale, data=dataset, goal=goal)
        #print(a.stateToRisk(10)) ### USE THIS TO GET RISK AT A STATE
        #print(a.indexToCoord)
        #print(a.coordToIndex)

    if True: 
        #MDP pipeline
        ### TRANSLATE DATASET TO MDP ###
        actions = {0: "right", 1: "up", 2: "left", 3: "down"}
        """
        P is A x S matrix, where P[a][s] is state obtained from performing action a in state s
        R(s,a) is reward obtained from performing action a from state s
        """
        P, R = state_dict_to_P(a.coordToIndex, a.stateToRisk, dataset.states, actions, f"mdp_params/{longitude}_{latitude}_{scale}/")
        print(P, R)

        goal_state = a.coordToIndex[goal[0]][goal[1]]

        ### SOLVE MDP using MDP toolbox ###
        ## label values: VI, PI, QL, SARSA
        label = "VI"
        ## VALUE ITERATION
        match label:
            case "VI":
                V, policy = value_iteration(P, R)
                label = "VI"
            case "PI":
                V, policy = policy_iteration(P, R)
                label = "PI"
            case "QL":
                V, policy = Q_learning(P, R, terminal_state=goal_state)
                label = "QL"
            case "SARSA":
                V, policy = SARSA(P, R, terminal_state=goal_state)
                label = "SARSA"

        save_result(policy, V, f"results/{longitude}_{latitude}_{scale}_{label}/")
    else:
        label = "QL"
        V, policy = read_result(f"results/{longitude}_{latitude}_{scale}_{label}")

    print(policy)
    print(is_valid_policy(policy, a.indexToCoord, dataset.states)) 

    if True:
        ### EVALUATE POLICY ###
        path = runPath(policy=policy, start=start, goal=goal, coordToIndex=a.coordToIndex, scale=scale)
        evaluator = Evaluator(scale, epochs=5, epoch_duration=30)
        print("Path score: ", evaluator.evalPolicy(path))
        coordToPolicy(a.coordToIndex, policy)

    if False: #example on how to do the functional approximation, verified and working
        environment=Environement("dataset_1")
        environment.encode(environment.dataset.goal, one_hot_encoding=True) #initialization
        environment.set_model()
        environment.train()
        policy=environment.generate_policy(seperate=True) #if not true 
        print(policy) #a coord:action policy as we discussed, ready to be printed

    if True:
        ### VISUALIZE DATASET ###
        #plot_dataset_on_map(dataset, Attribute="danger", Ranges=5)
        #plot_dataset_on_map(dataset, Attribute="density", Ranges=5) #- working as intended 


        ### VISUALIZE POLICY ###
        dataset_solved=read_dataset("functional_approximation_solved_demo")
        plot_dataset_on_map(dataset_solved, Attribute="action", Ranges=3)

        #the animation we discussed

main()