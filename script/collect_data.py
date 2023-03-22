import gym
import numpy as np

from cpprb import ReplayBuffer
from tqdm import tqdm
from safe_rl.policy import CVPO
from safe_rl.util.logger import EpochLogger
import torch
from safe_rl.util.torch_util import export_device_env_variable, seed_torch


def main(config):
    seed=0
    device="cpu"
    device_id=0
    threads=2
    seed_torch(seed)
    torch.set_num_threads(threads)
    export_device_env_variable(device, id=device_id)

    total_timesteps = config["total_timesteps"]
    env = gym.make(config["env"])

    model_dir = config["model_dir"]
    dummy_logger = EpochLogger(output_dir="data/test")
    expert = CVPO(env, dummy_logger)
    expert.load_model(model_dir)

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
    buffer_size = config["total_timesteps"]
    cpp_buffer = ReplayBuffer(buffer_size, env_dict)

    obs = env.reset()
    for i in tqdm(range(total_timesteps)):
        action, _ = expert.act(obs, 
                               deterministic=True,
                               with_logprob=False)
        obs_next, reward, done, info = env.step(action)
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if i == total_timesteps - 1 or "TimeLimit.truncated" in info else done
        # ignore the goal met terminal condition
        terminal = done
        done = True if "goal_met" in info and info["goal_met"] else done

        if "cost" in info:
            cost = info["cost"]
            cpp_buffer.add(obs=obs,
                                act=np.squeeze(action),
                                rew=reward,
                                obs2=obs_next,
                                done=done,
                                cost=cost)
        else:
            cpp_buffer.add(obs=obs,
                                act=np.squeeze(action),
                                rew=reward,
                                obs2=obs_next,
                                done=done)

        if terminal:
            obs = env.reset()
        else:
            obs = obs_next

        cpp_buffer.save_transitions(config["expert_data_dir"])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-PointButton1-v0')
    parser.add_argument('--total_timesteps', '-t', type=int, default=int(1e5))
    config = parser.parse_args()
    config = vars(config)

    model_dirs = {
        "Safexp-CarGoal1-v0": 
            "data/Safexp-CarGoal1-v0_cost_10/cvpo_jp_epoch_150_load_critic/" +
            "cvpo_jp_epoch_150_load_critic_s0/model_save/model.pt",
        "Safexp-CarButton1-v0":
            "data/Safexp-CarButton1-v0_cost_10/cvpo_jp_epoch_150_load_critic/" + 
            "cvpo_jp_epoch_150_load_critic_s0/model_save/model.pt",
        "Safexp-CarPush1-v0":
            "data/Safexp-CarPush1-v0_cost_10/cvpo_jp_epoch_150_load_critic/" +
            "cvpo_jp_epoch_150_load_critic_s0/model_save/model.pt",
        "Safexp-CarGoal2-v0":
            "data/Safexp-CarGoal2-v0_cost_10/cvpo_epoch_150_load_critic/" +
            "cvpo_epoch_150_load_critic_s0/model_save/model.pt",
        "Safexp-CarButton2-v0":
            "data/Safexp-CarButton2-v0_cost_10/cvpo_jp_epoch_150_load_critic/" +
            "cvpo_jp_epoch_150_load_critic_s0/model_save/model.pt",
        "Safexp-CarPush2-v0":
            "data/Safexp-CarPush2-v0_cost_10/cvpo_jp_epoch_150_load_critic/" +
            "cvpo_jp_epoch_150_load_critic_s0/model_save/model.pt",
    }
    config["model_dir"] = model_dirs[config["env"]]
    config["expert_data_dir"] = "data/expert_data_" + config["env"]

    main(config)