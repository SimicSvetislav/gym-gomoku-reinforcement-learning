# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    id='Gomoku9x9-v1',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'board_size': 9,
    },
    nondeterministic=True,
)

register(
    id='Gomoku19x19-v1',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'board_size': 19,
    },
    nondeterministic=True,
)

'''
register(
    id='Gomoku19x19-v0',
    entry_point='gym_gomoku.envs:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'beginner'
        'board_size': 19,
    },
    nondeterministic=True,
)
'''