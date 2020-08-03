# -*- coding: utf-8 -*-

import gym
import numpy as np
from dueling_deep_q_learning import GomokuAgent, GomokuAgentConvnet, plot_training
from tqdm import tqdm 


LAYERS = 2
DENSE_LAYER_NEURONS = 128
GAMES_NUM = 10000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
STOP_DECAY_PERCENT = 0.7

OPPONENT_TYPES = {'beginner': 0, 'random': 1}

BOARD_SIZE = 9
OPPONENT = 'beginner'

MODEL = 'model_conv'

ATTEMPT_NAME = f'pattern_opponent_11_{LAYERS}x{DENSE_LAYER_NEURONS}_{OPPONENT}_g{GAMES_NUM}_lr{LEARNING_RATE}_b{BATCH_SIZE}_s{STOP_DECAY_PERCENT}'

    
if __name__=='__main__':
    
    env = gym.make(f'Gomoku{BOARD_SIZE}x{BOARD_SIZE}-v{OPPONENT_TYPES[OPPONENT]}')
    agent = GomokuAgentConvnet(learning_rate=LEARNING_RATE, discount=0.99, 
                        actions_num=BOARD_SIZE**2, epsilon=0.0, 
                        batch_size=BATCH_SIZE, memory_size=100_000,
                        model_filename=f'gomoku_d3qn_{ATTEMPT_NAME}',
                        stop_decay_after=GAMES_NUM*STOP_DECAY_PERCENT,
                        epsilon_min=0.01, update_every=100, 
                        layers=LAYERS, neurons=DENSE_LAYER_NEURONS,
                        architecture=MODEL,
                        board_size=BOARD_SIZE,
                        # input_shape=[BOARD_SIZE**2]
                        input_shape=(BOARD_SIZE, BOARD_SIZE, 1),
                        trained_model='gomoku_d3qn_pattern_opponent_1_2x128_random_g10000_lr0.001_b128_s0.7_weights.npy',
                        # trained_model=None
                        )
    
    scores = []
    epsilon_history = []
    
    for i in tqdm(range(GAMES_NUM), ascii=True, unit='game'):
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(np.expand_dims(state, axis=2))
            # action = agent.choose_action(state.flatten())
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(np.expand_dims(state, axis=2), action, reward, np.expand_dims(new_state, axis=2), done)
            # agent.store_transition(state.flatten(), action, reward, new_state.flatten(), done)
            state = new_state
            agent.train()
        epsilon_history.append(agent.epsilon)
        
        scores.append(score)
        
        moving_average = np.mean(scores[-100:])
        print('episode: ', i+1, 'score: %.2f' % score, 
              ' average score %.2f' % moving_average,
              ' epsilon %.3f' % agent.epsilon)
        
        # agent.epsilon = max(agent.epsilon*agent.epsilon_decay_rate, agent.epsilon_min)
        
    agent.save_model()
        
    img_filename = f'gomoku_dueling_ddqn_ma100_{ATTEMPT_NAME}.png'
    
    x = [i+1 for i in range(GAMES_NUM)]
    
    plot_training(x, scores, epsilon_history, img_filename, OPPONENT, moving_avg_len=100)
