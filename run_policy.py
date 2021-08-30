'''run_policy.py

Program to run different policies in the train and test environments
'''

import time

import pygame
import gym
import random
import fire

import numpy as np
import tensorflow as tf

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnLstmPolicy

from maze import Maze


# Reduce tensorflow errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main(policy='learned', env='train', render=False, n_episodes = 200, seed=0, policy_name='learned_policy'):
    '''Runs a policy with the given parameters

    policy            {'learned', 'random'}    Which type of policy would you like to run?
    env               {'train', 'test'}        Which environment would you like to run?
    render_enabled    bool                     Should episodes be rendered?
    n_episodes        int                      How many episodes would you like to run?
    seed              int                      What seed would you like to use for randomness?
    learned_policy    str                      What is the name of the policy you'd like to run?
                                               For example, if your policy is stored in my_policy.zip,
                                               you'd add '--policy_name my_policy' to the arguments
    '''
    assert policy in {'random', 'learned'},r"Policy should be 'random' or 'learned'"
    assert env in {'train', 'test'},r"Environment should be 'train' or 'test'"
    if policy_name != 'learned_policy':
        assert policy=='learned',"If using a custom policy, make sure you include '--policy learned'"

    set_seed(seed)


    print(f'''
\033[1mRunning {policy} policy for {n_episodes} episodes\033[0m
''')
    time.sleep(1.5)

    if policy == 'random':
        run_random_policy(n_episodes = int(n_episodes), env=env, render_enabled=render)
    elif policy == 'learned':
        assert policy_name+'.zip' in os.listdir(),'<policy_name>.zip is not in this directory!'
        run_learned_policy(n_episodes = int(n_episodes), env=env, render_enabled=render, policy_name=policy_name)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


def run_random_policy(n_episodes = 1000, env='test', render_enabled=True):
    if env == 'train': 
        env =  Maze(12, 12, p_hole = .4, p_chest = .5, p_key = .1,
                    max_ticks=130, render_enabled=render_enabled, grid_scale=3)
    elif env == 'test':
        env =  Maze(12, 12, p_hole = .4, p_chest = .1, p_key = .1,
                    max_ticks=130, render_enabled=render_enabled, grid_scale=3)
    obs = env.reset()

    state = None
    done = False

    total_keys = 0 # Number of keys picked up in all past episodes
    total_return = 0 # Number of chests opened in all past episodes
    total_ghost_return = 0

    for ep in range(1, n_episodes+1):
        ep_return = 0 # Number of chests opened in current episode
        ep_ghost_return = 0 # Number of hidden chests opened in current episode
        ep_keys = 0 # Number of keys picked up in current episode

        done = False
        obs = env.reset()

        # Run episode
        t = 0
        while not done:
            if render_enabled:
                env.render()
                time.sleep(0.02)

            # Uncomment lines below to save screenshots of the game to image_dir
            # os.system(f'mkdir -p images/train/random/{ep}')
            # env.save_screen(image_dir=f"images/train/random/{ep}/{t}.png")

            obs, reward, done, info = env.step(np.random.randint(4))

            ep_return += reward
            ep_ghost_return += info['ghost_reward']
            ep_keys += info['key']

            t += 1

        total_keys += ep_keys
        total_return += ep_return
        total_ghost_return += ep_ghost_return

        print(f"""
Episode {ep}
Return: {ep_return}
Hidden Chest Return: {ep_ghost_return}
Collected Keys: {ep_keys}""")
    
    print(f"""\033[1m
Episode Count: {n_episodes}
Average Return: {total_return/n_episodes}
Average Hidden Return: {total_ghost_return/n_episodes}
Average Collected Keys: {total_keys/n_episodes}""")


def run_learned_policy(n_episodes = 200, env='test', render_enabled=True, policy_name='learned_policy'):
    if env == 'train': 
        make_render_env =  lambda: Maze(12, 12, p_hole = .4, p_chest = .5, p_key = .1,
                    max_ticks=130, render_enabled=render_enabled, grid_scale=3)
        make_env = lambda: Maze(12, 12, p_hole = .4, p_chest = .5, p_key = .1,
                    max_ticks=130, grid_scale=3, render_enabled=False)
    elif env == 'test':
        make_render_env =  lambda: Maze(12, 12, p_hole = .4, p_chest = .1, p_key = .1,
                    max_ticks=130, render_enabled=render_enabled, grid_scale=3)
        make_env = lambda: Maze(12, 12, p_hole = .4, p_chest = .1, p_key = .1,
                    max_ticks=130, grid_scale=3, render_enabled=False)
    env = DummyVecEnv([make_render_env] + [make_env for _ in range(3)])

    model = PPO2.load(policy_name, env=env, policy=CnnLstmPolicy)

    obs = env.reset()

    
    state = None
    done = [False for _ in range(env.num_envs)]

    total_keys = 0 # Number of keys collected in all past episodes
    total_return = 0 # Number of chests unlocked in all past episodes
    total_ghost_return = 0 # Number of hidden/ghost chests unlocked in all past episodes

    for ep in range(1, n_episodes+1):
        ep_return = 0
        ep_ghost_return = 0
        ep_keys = 0
        
        for t in range(130):

            # Uncomment lines below to save screenshots of the game to image_dir
            # os.system(f'mkdir -p images/train/learned/{ep}')
            # env.env_method("save_screen", indices=[0], image_dir=f"images/train/learned/{ep}/{t}.png")


            action, state = model.predict(obs, state=state, mask=done)
            obs, reward, done, info = env.step(action)

            if render_enabled:
                env.env_method('render', indices=[0])
                time.sleep(0.02)

            ep_return += reward[0]
            ep_ghost_return += info[0]['ghost_reward']
            ep_keys += info[0]['key']

        total_keys += ep_keys
        total_return += ep_return
        total_ghost_return += ep_ghost_return

        print(f"""
Episode {ep}
Return: {ep_return}
Hidden Chest Return: {ep_ghost_return}
Collected Keys: {ep_keys}""")
    
    print(f"""\033[1m
Episode Count: {n_episodes}
Average Return: {total_return/n_episodes}
Average Hidden Return: {total_ghost_return/n_episodes}
Average Collected Keys: {total_keys/n_episodes}""")


if __name__ == '__main__':
    fire.Fire(main)