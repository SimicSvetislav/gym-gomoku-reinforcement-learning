# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import math

class DDQN(Model):
    
    def __init__(self, actions_num, layer1_neurons, layer2_neurons):
        super(DDQN, self).__init__()
        
        self.dense1 = Dense(layer1_neurons, activation='relu')
        self.dense2 = Dense(layer2_neurons, activation='relu')
        
        self.value = Dense(1, activation=None)
        self.advantage = Dense(actions_num, activation=None)
        
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        
        V = self.value(x)
        A = self.advantage(x)
        
        Q = V + A - tf.math.reduce_mean(A, axis=1, keepdims=True)
        
        return Q
    
    def get_advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        
        A = self.advantage(x)
        
        return A


class DDQN_v2(Model):
    
    def __init__(self, actions_num, layers, neurons):
        super(DDQN_v2, self).__init__()
        
        self.dense_layers = []
        
        for i in range(layers):
            self.dense_layers.append(Dense(neurons, activation='relu'))
        
        self.value = Dense(1, activation=None)
        self.advantage = Dense(actions_num, activation=None)
        
    def call(self, state):
        
        x = state
        
        for layer in self.dense_layers:
            x = layer(x)
        
        V = self.value(x)
        A = self.advantage(x)
        
        Q = V + A - tf.math.reduce_mean(A, axis=1, keepdims=True)
        
        return Q
    
    def get_advantage(self, state):
        
        x = state
        
        for layer in self.dense_layers:
            x = layer(x)
        
        A = self.advantage(x)
        
        return A

class DDQN_convnet(Model):
    
    def __init__(self, actions_num, fc_layers, neurons, input_shape):
        super(DDQN_v2, self).__init__()
        
        
        self.conv_11 = Conv2D(256, (5,5), padding='same', 
                              input_shape=input_shape, activation='relu')
        self.conv_12 = Conv2D(256, (5,5), padding='same', activation='relu')
        self.pool_1 = MaxPooling2D(2,2)
        self.drop_1 = Dropout(0.2)
        
        self.conv_21 = Conv2D(256, (5,5), padding='same', activation='relu')
        self.conv_22 = Conv2D(256, (5,5), padding='same', activation='relu')
        self.pool_2 = MaxPooling2D(2,2)
        self.drop_2 = Dropout(0.2)
        
        self.flatten = Flatten()
        
        self.dense_layers = []
        
        for i in range(fc_layers):
            self.dense_layers.append(Dense(neurons, activation='relu'))
        
        self.value = Dense(1, activation=None)
        self.advantage = Dense(actions_num, activation=None)
        
    def call(self, state):
        
        x = self.conv_11(state)
        x = self.conv_12(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        
        x = self.conv_21(x)
        x = self.conv_22(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        
        x = self.flatten(x)
        
        for layer in self.dense_layers:
            x = layer(x)
        
        V = self.value(x)
        A = self.advantage(x)
        
        Q = V + A - tf.math.reduce_mean(A, axis=1, keepdims=True)
        
        return Q
    
    def get_advantage(self, state):
        
        x = self.conv_11(state)
        x = self.conv_12(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        
        x = self.conv_21(x)
        x = self.conv_22(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        
        x = self.flatten(x)
        
        for layer in self.dense_layers:
            x = layer(x)
        
        A = self.advantage(x)
        
        return A


class ReplayMemory():
    
    def __init__(self, max_size, input_shape):
        self.size = max_size
        self.counter = 0

        self.states = np.zeros((self.size, *input_shape))
        self.new_states = np.zeros((self.size, *input_shape))
        self.actions = np.zeros(self.size)
        self.rewards = np.zeros(self.size)
        self.dones = np.zeros(self.size)
        
    def save_transition(self, state, action, reward, new_state, done):
        index = self.counter % self.size
        
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.dones[index] = done
        
        self.counter += 1
        
    def sample(self, batch_size):
        occupied = min(self.counter, self.size)
        batch_indexes = np.random.choice(occupied, batch_size, replace=False)
        
        states_batch = self.states[batch_indexes]
        actions_batch = self.actions[batch_indexes]
        rewards_batch = self.rewards[batch_indexes]
        new_states_batch = self.new_states[batch_indexes]
        dones_batch = self.dones[batch_indexes]
        
        return states_batch, actions_batch, rewards_batch, new_states_batch, dones_batch
    
    
class Agent():
    
    def __init__(self, learning_rate, discount, actions_num, epsilon, batch_size,
                 input_shape, epsilon_decay_rate, epsilon_min, memory_size,
                 update_every, model_filename, layer1_neurons, layer2_neurons):
        
        self.action_space = [i for i in range(actions_num)]
        
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        
        self.update_every = update_every
        self.batch_size = batch_size
        
        self.learn_step_counter = 0
        
        self.memory = ReplayMemory(memory_size, input_shape)
        
        self.main_model = DDQN(actions_num, layer1_neurons, layer2_neurons)
        self.target_model = DDQN(actions_num, layer1_neurons, layer2_neurons)
        
        self.main_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                loss='mse')
        self.target_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                  loss='mse')
        
        self.model_filename = model_filename
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.save_transition(state, action, reward, new_state, done)
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # TODO: Izbaciti zauzete
            action = np.random.choice(self.action_space)
        else:
            state = np.array([state])
            actions = self.main_model.get_advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            
        return action
        
    def train(self):
        if self.memory.counter < self.batch_size:
            return
        
        if self.learn_step_counter % self.update_every == 0:
            # print("Get weigths:", len(self.target_model.get_weights()))
            print("Get weigths len:", len(self.target_model.get_weights()))
            print("Get weigths len:", len(self.main_model.get_weights()))
            self.target_model.set_weights(self.main_model.get_weights())
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        q_pred = self.main_model(states)
        q_next = self.target_model(new_states)
        
        # q_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        
        # q_target = np.copy(q_pred)
        q_target = q_pred.numpy()
        
        max_actions = tf.math.argmax(self.main_model(new_states), axis=1)
        
        for i, done in enumerate(dones):
            # if done:
            #    q_next[i] = 0
            
            # print(q_target.shape)
            q_target[i, int(actions[i])] = rewards[i] + self.discount*q_next[i, max_actions[i]]*(1 - int(dones[i]))
            # q_target[i, actions[i]] = rewards[i] + self.discount*q_next[i, max_actions[i]]*(1 - int(dones[i]))
            # q_target[i, actions[i]] = rewards[i] + self.discount*q_next[i]
            
        self.main_model.train_on_batch(states, q_target)
        
        # self.epsilon = self.epsilon * self.epsilon_decay_rate if self.epsilon > self.epsilon_min else self.epsilon_min
        # self.epsilon = self.epsilon - self.epsilon_decay_rate if self.epsilon > self.epsilon_min else self.epsilon_min
        
        self.learn_step_counter += 1
        
    def save_model(self):
        self.main_model.save(self.model_filename)
        
    def load_model(self):
        self.main_model = load_model(self.model_file)

        
class GomokuAgent():
    
    def __init__(self, learning_rate, discount, actions_num, epsilon, batch_size,
                 input_shape, stop_decay_after, epsilon_min, memory_size,
                 update_every, model_filename, layers, neurons,
                 trained_model=None):
    
        self.action_space = [i for i in range(actions_num)]
        
        self.discount = discount
        
        self.epsilon = epsilon
        # self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_rate = math.exp(math.log(epsilon_min)/stop_decay_after)
        self.epsilon_min = epsilon_min
        
        self.update_every = update_every
        self.batch_size = batch_size
        
        self.learn_step_counter = 0
        
        self.memory = ReplayMemory(memory_size, input_shape)
        
        # self.main_model = DDQN(actions_num, layer1_neurons, layer2_neurons)
        # self.target_model = DDQN(actions_num, layer1_neurons, layer2_neurons)
        
        # self.main_model = make_model('model_fc', actions_num=actions_num, neurons=layer1_neurons)
        # self.target_model = make_model('model_fc', actions_num=actions_num, neurons=layer1_neurons)
        
        if trained_model==None:
            self.main_model = make_model('model_fc_v2', layers=layers, actions_num=actions_num, neurons=neurons)
            self.target_model = make_model('model_fc_v2', layers=layers, actions_num=actions_num, neurons=neurons)
        else:
            load_model(trained_model)
            
        self.main_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                loss='mse')
        self.target_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                  loss='mse')
        
        self.model_filename = model_filename
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.save_transition(state, action, reward, new_state, done)
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            possible_actions = []
            for index, field_value in enumerate(state):
                if field_value==0:
                    possible_actions.append(index)
            action = np.random.choice(possible_actions)
        else:
            state = np.array([state])
            actions = self.main_model.get_advantage(state)
            actions = actions.numpy()
            # print("Actions :", actions)
            # print("State :", state)
            for index, field_value in enumerate(state[0]):
                if field_value!=0:
                    actions[0][index] = -math.inf
            # print(actions)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
            
        return action
        
    def train(self):
        
        # print('.', end='')
        
        if self.memory.counter < self.batch_size:
            return
        
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.update_every == 0:
            # print("Get weigths:", self.target_model.get_weights())
            # print("Get weigths len:", len(self.target_model.get_weights()))
            # print("Get weigths len:", len(self.main_model.get_weights()))
            self.target_model.set_weights(self.main_model.get_weights())
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        q_pred = self.main_model(states)
        q_next = self.target_model(new_states)
        
        # q_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        
        # q_target = np.copy(q_pred)
        q_target = q_pred.numpy()
        
        max_actions = tf.math.argmax(self.main_model(new_states), axis=1)
        
        for i, done in enumerate(dones):
            # if done:
            #    q_next[i] = 0
            
            q_target[i, int(actions[i])] = rewards[i] + self.discount*q_next[i, max_actions[i]]*(1 - int(dones[i]))
            
        self.main_model.train_on_batch(states, q_target)
        
    def save_model(self):
        self.main_model.save(self.model_filename)
        
    def load_model(self):
        self.main_model = load_model(self.model_file)
    
def plot_training(x, scores, epsilon_history, filename, lines=None, moving_avg_len=20):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    
    ax.plot(x, epsilon_history, color='C0')
    ax.set_xlabel('Game', color='C0')
    ax.set_ylabel('Epsilon', color='C0') 
    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')
    
    N = len(scores)
    running_avg = np.empty(N)
    for i in range(N):
        running_avg[i] = np.mean(scores[max(0,i-moving_avg_len):(i+1)])
        
    ax2.plot(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    
    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
            
    plt.savefig(filename)
    plt.show()

def make_model(model_name, **kwargs):
    if model_name=='model_fc':
        return DDQN(kwargs['actions_num'], kwargs['neurons'], kwargs['neurons'])
    elif model_name=='model_fc_v2':
        return DDQN_v2(kwargs['actions_num'], kwargs['layers'], kwargs['neurons'])
    elif model_name=='model_conv':
        return DDQN_convnet(kwargs['actions_num'], kwargs['layers'], 
                            kwargs['neurons'], kwargs['input_shape'])
    else:
        raise
