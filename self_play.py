# -*- coding: utf-8 -*-
import gym

env = gym.make('Gomoku9x9-v1')
env.reset()
env.render(mode='rgb_array')

def player_turn():
    
    while True:
        try:
            action = int(input('Type your action: '))
            break
        except ValueError:
            print("Action not valid!")
            continue
    
    return action

while True:
    # action = env.action_space.sample()
    
    action = player_turn()
    
    print(f'Put X on position {action}')
    
    try:
        observation, reward, done, info = env.step(action)
    except:
        print('Position already occupied!')
        continue
    
    env.render(mode='rgb_array')
    
    # print(observation)
    # print(info)
    # print(reward)
    
    if done:
        print ('Game is Over')
        
        if reward == 0.0:
            print('It\'s a draw!')
        elif reward == 1.0:
            print('You won!')
        elif reward == -1.0:
            print('You lost!')
            
        break