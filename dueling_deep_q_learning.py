# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce


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
        super(DDQN_convnet, self).__init__()
        
        
        self.conv_11 = Conv2D(128, (5,5), padding='same', 
                              input_shape=input_shape, activation='relu')
        self.conv_12 = Conv2D(128, (5,5), padding='same', activation='relu')
        self.pool_1 = MaxPooling2D(2,2)
        self.drop_1 = Dropout(0.2)
        
        self.conv_21 = Conv2D(64, (5,5), padding='same', activation='relu')
        self.conv_22 = Conv2D(64, (5,5), padding='same', activation='relu')
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
        
        array_shape = (self.size,) + input_shape
        
        self.states = np.zeros(array_shape, dtype=np.float32)
        self.new_states = np.zeros(array_shape, dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int32)
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

     
class GomokuAgent():
    
    def __init__(self, learning_rate, discount, actions_num, epsilon, batch_size,
                 input_shape, stop_decay_after, epsilon_min, memory_size,
                 update_every, model_filename, layers, neurons, architecture,
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
        
        input_shape = [reduce(lambda a, b: a*b, input_shape)]
        self.memory = ReplayMemory(memory_size, input_shape)
        
        if trained_model==None:
            self.main_model = make_model(architecture, layers=layers, 
                                         actions_num=actions_num, 
                                         neurons=neurons, 
                                         input_shape=input_shape)
            self.target_model = make_model(architecture, layers=layers, 
                                           actions_num=actions_num, 
                                           neurons=neurons,
                                           input_shape=input_shape)
        else:
            load_model(trained_model)
        
        self.main_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                loss='mse')
        self.target_model.compile(optimizer=Adam(learning_rate=learning_rate),
                                  loss='mse')
        
        self.model_file = trained_model
        
        
        
        
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

            for index, field_value in enumerate(state[0]):
                if field_value!=0:
                    actions[0][index] = -math.inf

            action = tf.math.argmax(actions, axis=1).numpy()[0]
            
        return action
    
    def train(self):    
        if self.memory.counter < self.batch_size:
            return
        
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.update_every == 0:
            self.target_model.set_weights(self.main_model.get_weights())
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        predicted_q_values = self.main_model(states).numpy()
        next_q_values = self.target_model(new_states)
        
        target_q_values = np.copy(predicted_q_values)
        
        max_actions = tf.math.argmax(next_q_values, axis=1)
        
        for i in range(len(states)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # max_next_q = np.max(next_q_values[i])
                # target_q_values[i, actions[i]] = rewards[i] + self.discount*max_next_q
                
                target_q_values[i, actions[i]] = rewards[i] + self.discount*next_q_values[i, max_actions[i]]
        
        self.main_model.train_on_batch(states, target_q_values)
        
    def save_model(self):
        self.main_model.save(self.model_filename)
        
    def load_model(self):
        self.main_model = load_model(self.model_file)

class GomokuAgentConvnet():
    
    def __init__(self, learning_rate, discount, actions_num, epsilon, batch_size,
                 input_shape, stop_decay_after, epsilon_min, memory_size,
                 update_every, model_filename, layers, neurons, architecture,
                 board_size, trained_model=None):
    
        self.action_space = [i for i in range(actions_num)]
        self.board_size = board_size
        
        self.discount = discount
        
        self.epsilon = epsilon
        # self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_decay_rate = math.exp(math.log(epsilon_min)/stop_decay_after)
        self.epsilon_min = epsilon_min
        
        self.update_every = update_every
        self.batch_size = batch_size
        
        self.learn_step_counter = 0
        
        self.memory = ReplayMemory(memory_size, input_shape)
        self.main_model = make_model(architecture, layers=layers, 
                                     actions_num=actions_num, 
                                     neurons=neurons, 
                                     input_shape=input_shape)
        self.target_model = make_model(architecture, layers=layers, 
                                       actions_num=actions_num, 
                                       neurons=neurons,
                                       input_shape=input_shape)        
        
        self.model_file = trained_model
        
        if trained_model!=None:
            self.main_model(tf.ones((1, *input_shape)))
            self.target_model(tf.ones((1, *input_shape)))

            self.load_trained_model()

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
            for i, row in enumerate(state):
                for j, cell in enumerate(row):
                    if cell==0:
                        possible_actions.append(i*self.board_size+j)
            action = np.random.choice(possible_actions)
        else:
            state = np.array([state], dtype=np.int32)
            actions = self.main_model.get_advantage(state.astype(np.float32))
            actions = actions.numpy()

            for i, row in enumerate(state[0]):
                for j, cell in enumerate(row):
                    if cell!=0:
                        actions[0][i*self.board_size+j] = -math.inf

            action = tf.math.argmax(actions, axis=1).numpy()[0]
            
        return action
    
    def train(self):    
        if self.memory.counter < self.batch_size:
            return
        
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.update_every == 0:
            self.target_model.set_weights(self.main_model.get_weights())
        
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        
        predicted_q_values = self.main_model(states).numpy()
        next_q_values = self.target_model(new_states)
        
        target_q_values = np.copy(predicted_q_values)
        
        max_actions = tf.math.argmax(next_q_values, axis=1)
        
        for i in range(len(states)):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # max_next_q = np.max(next_q_values[i])
                # target_q_values[i, actions[i]] = rewards[i] + self.discount*max_next_q
                
                target_q_values[i, actions[i]] = rewards[i] + self.discount*next_q_values[i, max_actions[i]]
        
        self.main_model.train_on_batch(states, target_q_values)
        
    def save_model(self):
        # self.main_model.save(self.model_filename)
        
        weights = self.main_model.get_weights()
        np.save(self.model_filename + '_weights', weights)
        
    def load_trained_model(self):
        # self.main_model = load_model(self.model_file)
        weights = np.load(self.model_file, allow_pickle=True)
        self.main_model.set_weights(weights)
        self.target_model.set_weights(weights)

def plot_training(x, scores, epsilon_history, filename, opponent, moving_avg_len=20):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    
    ax.plot(x, epsilon_history, color='C0')
    opponent = opponent if opponent=='random' else 'pattern-based'
    ax.set_title(f"vs opponent with {opponent} tactic")
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
