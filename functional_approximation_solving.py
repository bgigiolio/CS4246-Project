from tensorflow import keras
import numpy as np
import tensorflow as tf
from dataset import read_dataset
import random

#given: the full MDP, A x S matrix for transistions

#at least until the MDP saving is done using dataset directly, more flexible to learn a different function

#kanske Blakes reward function bra importera, index to cord med, so not same thing at two places

#De har inte thought about terminal state, so free to handle it my way

#NOT PRIORITZED!

class Environement:
    "class used for doing the deep Q learning, what action to take, the reward function calculation and so on. Fully based on the Chat-GPT solution but adopted to this case"
    def __init__(self, dataset_name, reward_function=lambda state: -state['to_goal']-state['danger']):
        self.dataset = read_dataset(dataset_name)
        self.state_space_size = len(self.dataset.states.keys())
        self.action_space_size = 4 #4 actions/state
        self.model = None #later, what to fit
        self.replay_buffer = [] #what experiences to replay everytime
        self.target_network = None
        self.reward_function=reward_function

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
            keras.layers.Dense(32, input_shape=(self.state_space_size,), activation='relu'), #state space size, 32 neurons, 32 neurons, 4 different actions... mapping state to action
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
            q_values = self.model.predict(state[np.newaxis]) #if not the random action take the action which is greedy according to the current state of the model
            action = np.argmax(q_values[0]) #np.newaxis increases the dimension by one
        return action

    def _update_epsilon(self):
        "make the odds of taking a random action lower over time"
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def _update_replay_buffer(self, experience):
        "???, add an experience for replay"
        self.replay_buffer.append(experience)
        
    def _train_q_network(self):
            if len(self.replay_buffer) >= self.batch_size:
                # Sample batch from replay buffer
                batch = np.array(random.sample(self.replay_buffer, self.batch_size)) #sample self.batch_size from the replay buffer which is all the transistions we have ever made!
                states, actions, rewards, next_states, dones = batch.T #transposes
                
                # Compute target Q-values
                next_q_values = self.target_network.predict(next_states)
                max_next_q_values = np.max(next_q_values, axis=1)
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
                
                # Compute predicted Q-values
                q_values = self.q_network.predict(states)
                q_values[np.arange(self.batch_size), actions] = target_q_values
                
                # Train Q-network on batch
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(self.loss_fn(q_values, self.q_network(states)))
                grads = tape.gradient(loss, self.q_network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
                
                # Update target network
                self.target_network.set_weights(self.q_network.get_weights())
        
    def run_episode(self, max_actions_to_take:int):
            def take_action(self, action):
                return next_state, reward, done
            state = self.dataset.start
            i=0
            done=False
            while not done or i>max_actions_to_take:
                # Choose action
                action = self._choose_action(state)
                
                # Take action
                next_state, reward, done = take_action(action)
                experience = (state, action, reward, next_state, done) #the experience gained
                self._update_replay_buffer(experience) #adding what we saw this time around
                
                # Update epsilon
                self._update_epsilon()
                
                # Update state
                state = next_state
                
                # Train Q-network - on everything and 
                self._train_q_network()

def main():
    print(1)

if __name__=='__main__':
    main()

