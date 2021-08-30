'''play.py

Lets humans (like you, probably!) play the train and test environments.
'''

import pygame
import gym
import fire

from maze import Maze


def play(env='test', n_episodes=1):
    '''Function that lets you (human) play the environment

    env           {'train', 'test'}     'train' is the environment the agent was trained on,
                                          'test' is the environment the agent was tested on.
    n_episodes    int                   How many episodes would you like to run?
    '''
    assert env in {'train', 'test'},"env should be 'train' or 'test'"

    if env == 'train':
        env = Maze(12, 12, p_hole = .4, p_chest = .5, p_key = .1, max_ticks=130)
    elif env == 'test':
        env = Maze(12, 12, p_hole = .4, p_chest = .1, p_key = .1, max_ticks=130)

    

    total_keys = 0 # Number of keys collected in all past episodes
    total_return = 0 # Number of chests unlocked in all past episodes
    total_ghost_return = 0 # Number of hidden/ghost chests unlocked in all past episodes

    for ep in range(1, int(n_episodes)+1):

        obs = env.reset()
        done = False

        ep_return = 0
        ep_ghost_return = 0
        ep_keys = 0

        while not done:
            action = env.get_human_action()
            obs, reward, done, info = env.step(action)
            
            ep_return += reward
            ep_ghost_return += info['ghost_reward']
            ep_keys += info['key']

            env.render()

        total_keys += ep_keys
        total_return += ep_return
        total_ghost_return += ep_ghost_return

        print(f"""\033[1m
Episode {ep}
Return: {ep_return}
Hidden Chest Return: {ep_ghost_return}
Collected Keys: {ep_keys}\033[0m""")

    print(f"""\033[1m
Average Return: {total_return/ep}
Average Hidden Chest Return: {total_ghost_return/ep}
Average Collected Keys: {total_keys/ep}
\033[0m""")



if __name__ == '__main__':
    fire.Fire(play)