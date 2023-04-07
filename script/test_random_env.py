import safety_gym
import gym
import numpy as np

env = gym.make("Safexp-CarButton1-v0")
env.seed(4)
env.set_num_different_layouts(2)

last_obs1 = env.reset()
env.step(env.action_space.sample())

last_obs2 = env.reset()
env.step(env.action_space.sample())

for i in range(10):
    obs = env.reset()
    env.step(env.action_space.sample())
    assert np.abs(np.sum(obs-last_obs1)) < 1e-6
    last_obs1 = obs


    obs = env.reset()
    env.step(env.action_space.sample())
    assert np.abs(np.sum(obs-last_obs2)) < 1e-6
    last_obs2 = obs

    print(np.sum(last_obs1 - last_obs2))

env.close()
