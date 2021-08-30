# Mesa Misalignment Environment

An environment for producing mesa misalignment, originally proposed by Matthew Barnett [here](https://www.lesswrong.com/posts/AFdRGfYDWQqmkdhFq/a-simple-environment-for-showing-mesa-misalignment). For an introduction to the concept of mesa-misalignment, see [here](https://arxiv.org/abs/1906.01820) and/or [here](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB).

The environment is a maze/gridworld. Every time the player lands on a key, it is automatically picked up. More than one key can be stored at a time. If the player steps on a chest and has at least one key, the chest is open, a key is lost, and the player gets +1 reward.

There are two variants of this environment, the 'train' environment and the 'test' environment.

In the training environment there are many chests and few keys, so once a player picks up a key they'll most often step on a chest tile without needing to pathfind to one. This means that a reinforcement learning agent doesn't need to learn to go towards chests in order to succeed in the training environment, it only needs to learn the proxy task of pathfind to keys.

I've used a [stable baseline](https://github.com/hill-a/stable-baselines) implementation of [PPO](https://arxiv.org/abs/1707.06347) to train a policy in the test environment. I then check performance in the test and train environments.

To see whether or not the policy is 'trying' to pick up chests, I put hidden chests in the environment that are distributed in the same way as the visible chests, except that they are hidden from the policy.

![](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FNoa%2FdIUZvU8vJ6.png?alt=media&token=89e45296-5f01-4cbf-9488-aefb7d11f0e5)

Data: https://docs.google.com/spreadsheets/d/1hfeJqsCK861Ff7Ne-KNPhf0_Rv3rgn8POzaQXn00hrw/edit?usp=sharing


To use this repository, first [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Then run

```conda env -n mesa-misalignment-example python=3.7```

```conda activate mesa-misalignment-example```

```pip install -r cpu-requirements.txt```

OR 

```pip install -r gpu-requirements.txt```

if you're using a GPU.

Use `python <file>.py --help` for information on how to play the environment yourself (`play.py`), run a policy for some number of episodes to collect data (`run_policy.py`), or replicate the training process (`train_policy.py`).

Thank you to 0x72 for almost all of the art! It was downloaded from [here](https://0x72.itch.io/dungeontileset-ii).