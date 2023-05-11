import gym
import safety_gym
import numpy as np
from cpprb import ReplayBuffer


env_name = "Safexp-CarButton1-v0"
seed = 2
data_dir = "data/expert_data_" + env_name + "_s" + str(seed) + ".npz"

env = gym.make(env_name)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape
env_dict = {
    'act': {
        'dtype': np.float32,
        'shape': act_dim
    },
    'done': {
        'dtype': np.float32,
    },
    'obs': {
        'dtype': np.float32,
        'shape': obs_dim
    },
    'obs2': {
        'dtype': np.float32,
        'shape': obs_dim
    },
    'rew': {
        'dtype': np.float32,
    }
}
if "Safe" in env.spec.id:
        env_dict["cost"] = {'dtype': np.float32}

buffer_size = 20000
cpp_buffer = ReplayBuffer(buffer_size, env_dict)

cpp_buffer.load_transitions(data_dir)

import pdb;pdb.set_trace()