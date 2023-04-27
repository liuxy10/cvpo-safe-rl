from cpprb import ReplayBuffer
import gym
import safety_gym
import numpy as np
import matplotlib.pyplot as plt

guide_data_dir = 'data/expert_data_Safexp-CarButton1-v0_s3.npz'
expl_data_dir = 'data/expert_data_Safexp-CarButton1-v0_s3_cvpo_jp.npz'

env_name = 'Safexp-CarButton1-v0'
env = gym.make(env_name)


obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
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

guide_cpp_buffer = ReplayBuffer(buffer_size, env_dict)
guide_cpp_buffer.load_transitions(guide_data_dir)

expl_cpp_buffer = ReplayBuffer(buffer_size, env_dict)
expl_cpp_buffer.load_transitions(expl_data_dir)


bins = 30

guide_obs = guide_cpp_buffer.get_all_transitions()['obs']
expl_obs = expl_cpp_buffer.get_all_transitions()['obs']

guide_act = guide_cpp_buffer.get_all_transitions()['act']
expl_act = expl_cpp_buffer.get_all_transitions()['act']

for i in range(act_dim):
    fig, axes = plt.subplots(
        figsize=(10, 5), ncols=2, nrows=1)
    axes = axes.flatten()
    axes[0].hist(
        guide_act[:, i],
        bins=bins,
        color='g',
        density=True,
        label=f'guide act {str(i)}',
    )
    axes[0].legend()
    axes[1].hist(
        expl_act[:, i],
        bins=bins,
        color='b',
        density=True,
        label=f'expl act {str(i)}',
    )
    axes[1].legend()
    plt.savefig('figures/dist/' + env_name + f'_act_{str(i)}.pdf', dpi=150)
    plt.close()

for i in range(obs_dim):
    fig, axes = plt.subplots(
        figsize=(10, 5), ncols=2, nrows=1)
    axes = axes.flatten()
    axes[0].hist(
        guide_obs[:, i],
        bins=bins,
        color='g',
        density=True,
        label=f'guide obs {str(i)}',
    )
    axes[0].legend()
    axes[1].hist(
        expl_obs[:, i],
        bins=bins,
        color='b',
        density=True,
        label=f'expl obs {str(i)}',
    )
    axes[1].legend()
    plt.savefig('figures/dist/' + env_name + f'_obs_{str(i)}.pdf', dpi=150)
    plt.close()

