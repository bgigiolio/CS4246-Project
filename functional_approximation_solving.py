from tensorflow import keras
import numpy as np
import tensorflow as tf
from dataset import read_dataset
import random
from buildRiskTable import MDP
import tqdm
from buildRiskTable import riskCalcNeighborsGoal #to be updated with density

#POSSIBLE WORK BUT MOST LIKELY NO TIME:
#TODO - extend the not one-hot-encoding, test so that it works, for example by adding the reward of states more steps ahead, using CNN to measure the form of danger in the area
#TODO - make it possible to use the not one-hot-encoding for solving globally. This requires generating the state, next state for that part on demand!
#TODO - vary stuff in the algoritm, can it be made even better?
#TODO - clear the buffer a bit?
#TODO - allways return 

class Environement:
    "class used for doing the deep RL, what action to take, the reward function calculation and so on. Fully based on the Chat-GPT solution but adopted to this case"
    def __init__(self, dataset_name):
        self.dataset = read_dataset(dataset_name)
        self.state_info_size = None #storleken of what we send in to the neural network, decided later
        self.action_space_size = 4 #4 actions/state
        self.model = None #later, what to fit
        self.replay_buffer = [] #what experiences to replay everytime
        self.target_network = None
        #not used currently
        self.mdp=MDP(lon=(self.dataset.min_lon, self.dataset.max_lon), lat=(self.dataset.min_lat, self.dataset.max_lat), scale=self.dataset.scale, data=self.dataset, goal=self.dataset.goal)

        #print("coordToIndex", self.mdp.coordToIndex) #detta borde fungera, jag hade fel

        #Random ML parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99 #decay by 1 % everytime
        self.min_epsilon = 0.01
        self.learning_rate = 0.001
        self.loss_fn = keras.losses.mean_squared_error #MSE error metric
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        
    def set_model(self, model=None):
        "set the model, has to be a tensorflow model"
        if not model:
            q_network = keras.Sequential([
            keras.layers.Dense(32, input_shape=(self.state_info_size,), activation='relu'), #state space size, 32 neurons, 32 neurons, 4 different actions... mapping state to action
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_space_size) #the last layer has 4 different outputs, one for each action
            ])
            self.model=q_network
            self.target_network=keras.models.clone_model(self.model)
            self.target_network.set_weights(self.model.get_weights())
        else:
            self.model=model
            self.target_network=keras.models.clone_model(self.model)
            self.target_network.set_weights(self.model.get_weights())
    
    def _choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_space_size) #take a random action from the action space size
        else:
            q_values = self.model.predict(state[np.newaxis], verbose=0) #if not the random action take the action which is greedy according to the current state of the model
            action = np.argmax(q_values[0]) #np.newaxis increases the dimension by one
        return action

    def _update_epsilon(self):
        "make the odds of taking a random action lower over time"
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def _update_replay_buffer(self, experience):
        "add an experience for replay (the sampling of data to train on)"
        self.replay_buffer.append(experience)
        
    def _train_q_network(self):
            "here only working with the encoded info about each state"
            if len(self.replay_buffer) >= self.batch_size: #until we have sampled the batch size don't do any training
                # Sample batch from replay buffer

                batch = random.sample(self.replay_buffer, self.batch_size) #sample self.batch_size from the replay buffer which is all the transistions we have ever made!

                #IF NOT WORKING THIS LIKELY IS THE ISSUE, OR TOO few iterations...
                states=np.array([batch[i][0] for i in range(len(batch))])
                next_states=np.array([batch[i][3] for i in range(len(batch))])
                actions=np.array([batch[i][1] for i in range(len(batch))])
                rewards=np.array([batch[i][2] for i in range(len(batch))])
                dones=np.array([batch[i][4] for i in range(len(batch))])
                
                # Compute target Q-values
                next_q_values = self.target_network.predict(next_states, verbose=0)
                max_next_q_values = np.max(next_q_values, axis=1)
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                
                # Compute predicted Q-values
                q_values = self.model.predict(states, verbose=0)
                q_values[np.arange(self.batch_size), actions] = target_q_values
                
                # Train Q-network on batch
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(self.loss_fn(q_values, self.model(states)))
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # Update target network
                self.target_network.set_weights(self.model.get_weights())
        
    def take_action(self, action, state):
        #print(state)
        neighbours=self.dataset.states[state]["neighbours"]
        next_state=None
        for (close_state, index) in neighbours:
            if index==action:
                next_state=close_state
        if not next_state:
            next_state=state #took an action which lead us back to same state
        
        (lon,lat)=next_state
        
        reward=riskCalcNeighborsGoal(lon, lat, self.dataset, self.dataset.goal) #only thing which is tabular is the danger

        if next_state==self.dataset.goal:
            done=True
        else:
            done=False
        return next_state, reward, done

    def encode(self, state, one_hot_encoding=True):
        "makes a numpy array containing the information about the state used for the neural network"
        
        if not one_hot_encoding: #the idea her is to find a rule based on the features of the state
            info_state=[]
            
            neighbours=self.dataset.states[state]["neighbours"]
            
            #adding distance to goal, danger for the neighbouring states as a starting hypothesis, for every action

            for action in range(self.action_space_size):
                next_state=None
                for (close_state, index) in neighbours:
                    if index==action:
                        next_state=close_state
                        info_state.append(self.dataset.states[next_state]["danger"])
                        info_state.append(self.dataset.states[next_state]["to_goal"])
                if not next_state:
                    info_state.append(self.dataset.states[state]["danger"])
                    info_state.append(self.dataset.states[state]["to_goal"])

            self.state_info_size=len(info_state)
            return np.array(info_state)
        
        else: #requires sampling the same space as the function approximation is tested on for this to work! hard
            info_state=[]
            for state_key in self.dataset.states.keys():
                if state==state_key:
                    info_state.append(1)
                else:
                    info_state.append(0)
            self.state_info_size=len(info_state)
            #print(self.state_info_size)
            return np.array(info_state)


    def run_episode(self, max_actions_to_take:int=100):
            state = self.dataset.start
            done=False #if reaches goal then done
            for _ in tqdm.tqdm(range(max_actions_to_take), desc="running episode but will break if reaches goal"):
                if done: #use the final place
                    break
                # Choose action
                action = self._choose_action(self.encode(state))

                # Take action
                next_state, reward, done = self.take_action(action, state)

                #one-hot-encoding by deafult
                experience = (self.encode(state), action, reward, self.encode(next_state), done) #the experience gained, translated to information which can be input to neural network
                self._update_replay_buffer(experience) #adding what we saw this time around
                
                # Update epsilon
                self._update_epsilon()
                
                # Update state
                state = next_state
                
                # Train Q-network - on everything we have seen
                self._train_q_network()

    def train(self, episodes_to_run:int=100):
        "fits the model by some episodes"
        for _ in tqdm.tqdm(range(episodes_to_run), desc="training the model by running some episodes"):
            self.run_episode()

    def generate_policy_utility(self):
        "to be done after training, returns a cord:action dictionary and a utility dictionary coord:value as well as add those attributes to the dictionary"

        policy={}
        utility={}
        states=self.states
        for state in tqdm.tqdm(self.dataset.states.keys(), desc="calculating policy"):
            state_encoded=self.encode(state)
            q_values = self.model.predict(state_encoded[np.newaxis], verbose=0) #if not the random action take the action which is greedy according to the current state of the model
            action = np.argmax(q_values[0])
            util=np.max(q_values[0])
            utility[state]=util
            policy[state]=action
            states[state]['action']=int(action)
            states[state]['utility']=float(util)
        self.dataset.states=states
        return policy, utility

def main():
    environment=Environement("dataset_1")
    environment.encode(environment.dataset.goal, one_hot_encoding=True) #initialization
    environment.set_model()
    environment.train(100)
    policy, utility=environment.generate_policy_utility()
    print(policy) #

if __name__=='__main__':
    main()

