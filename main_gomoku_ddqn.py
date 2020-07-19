# -*- coding: utf-8 -*-

import gym
import numpy as np
from dueling_deep_q_learning import GomokuAgent, plot_training
from tqdm import tqdm


LAYERS = 5
DENSE_LAYER_NEURONS = 256
GAMES_NUM = 5000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
STOP_DECAY_PERCENT = 0.7

OPPONENT_TYPES = {'beginner': 0, 'random': 1}

BOARD_SIZE = 9
OPPONENT = 'random'

ATTEMPT_NAME = f'{LAYERS}x{DENSE_LAYER_NEURONS}_{OPPONENT}_g{GAMES_NUM}_lr{LEARNING_RATE}_b{BATCH_SIZE}_s{STOP_DECAY_PERCENT}'

    
if __name__=='__main__':
    
    env = gym.make(f'Gomoku{BOARD_SIZE}x{BOARD_SIZE}-v{OPPONENT_TYPES[OPPONENT]}')
    agent = GomokuAgent(learning_rate=LEARNING_RATE, discount=0.99, 
                        actions_num=BOARD_SIZE**2, epsilon=1.0, 
                        batch_size=BATCH_SIZE, memory_size=100_000,
                        model_filename=f'gomoku_d3qn_{ATTEMPT_NAME}',
                        stop_decay_after=GAMES_NUM*STOP_DECAY_PERCENT,
                        epsilon_min=0.01, update_every=100, 
                        layers=LAYERS, neurons=DENSE_LAYER_NEURONS,
                        trained_model=None,
                        input_shape=[BOARD_SIZE**2]
                        # input_shape=(BOARD_SIZE, BOARD_SIZE, 1)
                        )
    
    scores = []
    epsilon_history = []
    moving_average_acc = []
    
    for i in tqdm(range(GAMES_NUM), ascii=True, unit='game'):
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state.flatten())
            new_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state.flatten(), action, reward, new_state.flatten(), done)
            state = new_state
            agent.train()
        epsilon_history.append(agent.epsilon)
        
        # print('Score:', score)
        scores.append(score)
        
        moving_average = np.mean(scores[-100:])
        moving_average_acc.append(moving_average)
        print('episode: ', i+1, 'score: %.2f' % score, 
              ' average score %.2f' % moving_average,
              ' epsilon %.3f' % agent.epsilon)
        
        agent.epsilon = max(agent.epsilon*agent.epsilon_decay_rate, agent.epsilon_min)
        
    agent.save_model()
        
    filename = f'gomoku_dueling_ddqn_{ATTEMPT_NAME}.png'
    filename_ma_100 = f'gomoku_dueling_ddqn_100_{ATTEMPT_NAME}.png'
        
    x = [i+1 for i in range(GAMES_NUM)]
    
    plot_training(x, scores, epsilon_history, filename)    
    plot_training(x, scores, epsilon_history, filename_ma_100, moving_avg_len=100)
    # plot_training(x, moving_average_acc, epsilon_history, filename_moving_avg)
