from tensorflow import keras
import numpy as np
import tensorflow as tf
from dataset import read_dataset
import random
import tqdm
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

def draw_paths(paths:list, environment):
    "draws some paths obtained when running episodes"

    margin = 2 # buffer to add to the range
    lat_min = environment.dataset.min_lat - margin
    lat_max = environment.dataset.max_lat + margin
    lon_min = environment.dataset.min_lon - margin
    lon_max = environment.dataset.max_lon + margin

    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
        llcrnrlat=lat_min,
        urcrnrlon=lon_max,
        urcrnrlat=lat_max,
        lat_0=(lat_max - lat_min)/2,
        lon_0=(lon_max-lon_min)/2,
        projection='merc',
        resolution = 'h', #high resolution
        area_thresh=10000.,
        )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color = 'white',lake_color='#46bcec') #color of oceans

    for path in paths:
        lons=[path[i][0] for i in range(len(path))]
        lats=[path[i][1] for i in range(len(path))]
        lons, lats = m(np.array(lons), np.array(lats)) #convert to map cordinates
        m.plot(lons, lats) 

    plt.show()

class Environement:
    "class used for doing the deep RL, what action to take, the reward function calculation and so on. state in the code means (lon,lat)"
    def __init__(self, dataset_name, one_hot_encoding):
        self.one_hot_encoding=one_hot_encoding #defining if one hot encoding or (x,y) relative to goal input
        self.dataset = read_dataset(dataset_name)
        self.state_info_size = None #initialized later
        self.action_space_size = 4 #4 directions/state
        self.replay_buffer = [] #what experiences to replay everytime
        self.model = None #later
        self.target_network = None

        self.batch_size = 64
        self.gamma = 0.99 #easier to learn if having this? stabilises a bit?
        self.learning_rate = 3e-4 #must be optimized 
        self.loss_fn = keras.losses.mean_squared_error #MSE error metric
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def set_model(self, model=None):
        "set the model, has to be a tensorflow model"
        if not model: 
            #might be good to increase this even more
            percept1=self.state_info_size*20
            percept2=self.state_info_size*20 
            q_network = keras.Sequential([ 
            keras.layers.Dense(percept1, input_shape=(self.state_info_size,), activation='tanh', kernel_initializer='random_normal', use_bias=False), 
            keras.layers.Dense(percept2, activation='tanh', kernel_initializer='random_normal', use_bias=False), 
            keras.layers.Dense(self.action_space_size, kernel_initializer='random_normal', use_bias=False) 
            ])
            self.model=q_network
            self.target_network=keras.models.clone_model(self.model)
            self.target_network.set_weights(self.model.get_weights())
        else:
            self.model=model
            self.target_network=keras.models.clone_model(self.model)
            self.target_network.set_weights(self.model.get_weights())
    

    def _choose_action(self, encoded_state, state, p_random):
        if np.random.rand() < p_random:
            action = np.random.randint(self.action_space_size) #take a random action from the action space size
        else:
            q_values = self.model.predict(encoded_state[np.newaxis], verbose=0)
            action = np.argmax(q_values[0]) 
        return action

    def _update_replay_buffer(self, experience):
        "add an experience for replay (the sampling of data to train on)"
        self.replay_buffer.append(experience)

    def _load_weights(self, filename:str='model_weights.h5'):
        self.model.load_weights(filename)
        
    def _train_q_network(self):
            if len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)

                enc_states=np.array([batch[i][0] for i in range(len(batch))])
                next_enc_states=np.array([batch[i][3] for i in range(len(batch))])
                actions=np.array([batch[i][1] for i in range(len(batch))])
                rewards=np.array([batch[i][2] for i in range(len(batch))])
                dones=np.array([batch[i][4] for i in range(len(batch))])
                
                # Compute target Q-values
                next_q_values = self.target_network.predict(next_enc_states, verbose=0)
                max_next_q_values = np.max(next_q_values, axis=1)
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                
                # Compute predicted Q-values
                q_values = self.model.predict(enc_states, verbose=0)
                q_values[np.arange(self.batch_size), actions] = target_q_values
                
                # Train Q-network on batch
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(self.loss_fn(q_values, self.model(enc_states)))
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                return np.mean(loss.numpy())
            return 0
        
    def take_action(self, action, state):
        neighbours=self.dataset.states[state]["neighbours"]
        next_state=None
        reward=0
        for (close_state, index) in neighbours: #verified and correct
            if index==action:
                next_state=close_state
        if not next_state:
            next_state=state #took an action which lead us back to same state
            #reward-=1 #Penalty for hitting islands

        reward+=-self.dataset.states[state]["to_goal"]
        
        if next_state==self.dataset.goal:
            done=True
            reward=10 #good to be in goal but should not tilt the brain of the robot
        else:
            done=False
        return next_state, reward, done

    def encode(self, state):
        "makes a numpy array containing the information about the state used for the neural network"
        
        if not self.one_hot_encoding:
            info_state=[]
            
            #relative location
            (lon,lat)=state
            (lon_goal, lat_goal)=self.dataset.goal
            info_state.append(lon-lon_goal)
            info_state.append(lat-lat_goal)

            #encoding of more stuff needed for more advanced reward functions only
            if False:
                neighbours=self.dataset.states[state]["neighbours"] 
                for action in range(self.action_space_size):
                    next_state=None
                    for (close_state, index) in neighbours:
                        if index==action:
                            next_state=close_state
                            info_state.append(self.dataset.states[next_state]["danger"])
                            info_state.append(self.dataset.states[next_state]["density"])
                            info_state.append(self.dataset.states[state]["to_goal"])
                    if not next_state: #there is no such neighbour
                        info_state.append(self.dataset.states[state]["danger"])
                        info_state.append(self.dataset.states[state]["density"])
                        info_state.append(self.dataset.states[state]["to_goal"])
        
        else: #requires sampling the same space
            info_state=[]
            for state_key in self.dataset.states.keys():
                if state==state_key:
                    info_state.append(1)
                else:
                    info_state.append(0)
        
        self.state_info_size=len(info_state)
        return np.array(info_state)


    def run_episode(self, max_actions_to_take:int=100, p_random=0.1, stochastic_start=False):
            
            #this part is very specific to the close to Singapore scenario and might not work otherwise
            randomness=random.random()
            if stochastic_start and randomness<=0.5:
                (lon,lat) = self.dataset.start
                state=(lon-0.5,-lat) 
            else:
                state = self.dataset.start 
            
            #an extra start position to sample
            #elif stochastic_start and randomness<0.67:
                #(lon,lat) = self.dataset.goal
                #state=(lon-2,lat+1) 
            
            done=False
            total_reward=0
            total_loss=0
            path=[]
            for _ in tqdm.tqdm(range(max_actions_to_take), desc="running episode but will break if reaches goal"):
                if done:
                    break

                # Choose action
                action = self._choose_action(self.encode(state), state, p_random)

                # Take action
                next_state, reward, done = self.take_action(action, state)

                total_reward+=reward

                #the experience gained, translated to information which can be input to neural network
                experience = (self.encode(state), action, reward, self.encode(next_state), done)
                self._update_replay_buffer(experience) #adding what agent saw
                
                # Update state
                state = next_state

                path.append(state)
                
                loss=self._train_q_network()
                total_loss+=loss

            return total_reward, path, done, total_loss
    
    def train(self, episodes_to_run:int=100, max_actions_per_episode=100, stochastic_start=False):
        "fits the model by some episodes"

        total_rewards=[]
        paths=[]
        dones=[]
        loses=[]

        for episode in tqdm.tqdm(range(episodes_to_run), desc="training the model by running some episodes"):
            p_random=min(0.95,max(episodes_to_run*0.01/max(episode,1),0.01)) #GLIE
            if episode>min(episodes_to_run-2, 0.96*episodes_to_run): 
                p_random=0 #in order to make sure have time to train
            total_reward, path, done, total_loss=self.run_episode(max_actions_per_episode, p_random, stochastic_start)
            
            self.optimizer = keras.optimizers.Adam(learning_rate=(1-episode/episodes_to_run)*self.learning_rate)

            ### IMPORTANT TO TUNE!!!
            if episode % 9 == 0:
                self.target_network.set_weights(self.model.get_weights())
            
            dones.append(100*done)
            total_rewards.append(total_reward)
            loses.append(total_loss)

            #the starting random path
            if episode==1:
                paths.append(path)

        #saving the weights so that the model can be used on a widely different scenario potentially!
        self.model.save_weights('model_weights.h5')

        #appending the final path for reference
        paths.append(path)

        return total_rewards, paths, dones, loses

    def generate_policy_utility(self):
        "to be done after training, returns a cord:action dictionary \
            and a utility dictionary coord:value as well as add those attributes states dictionary of the dataset"
        policy={}
        utility={}
        states=self.dataset.states
        for state in tqdm.tqdm(self.dataset.states.keys(), desc="calculating policy"):
            state_encoded=self.encode(state)
            q_values = self.model.predict(state_encoded[np.newaxis], verbose=0)
            action = np.argmax(q_values[0])
            util=np.max(q_values[0])
            utility[state]=util
            policy[state]=action
            states[state]['action']=int(action)
            states[state]['utility']=float(util)
        self.dataset.states=states #now updated with the solution in the dataset
        return policy, utility
