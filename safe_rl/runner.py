import time
from copy import deepcopy
import gym
import numpy as np
import torch
# from tqdm import tqdm
import tqdm
import json
# from memory_profiler import profile

from safe_rl.policy import DDPG, SAC, TD3, SACLagrangian, DDPGLagrangian, TD3Lagrangian, CVPO, BC, CVPOMQL, CVPOIQL, SACLagFixed
from safe_rl.util.logger import EpochLogger, setup_logger_kwargs
from safe_rl.util.run_util import load_config, setup_eval_configs
from safe_rl.util.torch_util import export_device_env_variable, seed_torch
from safe_rl.util import js_utils
from safe_rl.worker import OffPolicyWorker, OnPolicyWorker, JumpStartOffPolicyWorker

import os
import sys
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/testing")
from utils import AddCostToRewardEnv
from visualize import plot_waymo_vs_pred




try:
    import bullet_safety_gym
except ImportError:
    print("can not find bullet gym...")

try:
    import safety_gym
    print(safety_gym.__file__)
except ImportError:
    print("can not find safety gym...")


class Runner:
    '''
    Main entry that coodrinate learner and worker
    '''
    # First element is the policy class while the second is whether it is an on-policy algorithm
    POLICY_LIB = {
        "sac": (SAC, False, OffPolicyWorker),
        "sac_lag": (SACLagrangian, False, OffPolicyWorker),
        "td3": (TD3, False, OffPolicyWorker),
        "td3_lag": (TD3Lagrangian, False, OffPolicyWorker),
        "ddpg": (DDPG, False, OffPolicyWorker),
        "ddpg_lag": (DDPGLagrangian, False, OffPolicyWorker),
        "cvpo": (CVPO, False, OffPolicyWorker),
        "sac_lag_jp": (SACLagrangian, False, JumpStartOffPolicyWorker),
        "cvpo_jp": (CVPO, False, JumpStartOffPolicyWorker),
        "bc": (BC, False, OffPolicyWorker),
        "cvpo_mql_jp": (CVPOMQL, False, JumpStartOffPolicyWorker),
        "cvpo_iql_jp": (CVPOIQL, False, JumpStartOffPolicyWorker),
        "cvpo_iql": (CVPOIQL, False, OffPolicyWorker),
        "sac_lag_fixed": (SACLagFixed, False, OffPolicyWorker),
    }

    def __init__(self,
                 sample_episode_num=50,
                 episode_rerun_num=10,
                 evaluate_episode_num=1,
                 mode="train",
                 exp_name="exp",
                 seed=0,
                 device="cpu",
                 device_id=0,
                 threads=2,
                 policy="ddpg",
                 env="Pendulum-v0",
                 timeout_steps=200,
                 epochs=10,
                 save_freq=20,
                 pretrain_dir=None,
                 load_dir=None,
                 data_dir=None,
                 verbose=True,
                 **kwarg) -> None:
        seed_torch(seed)
        torch.set_num_threads(threads)
        export_device_env_variable(device, id=device_id)

        self.episode_rerun_num = episode_rerun_num
        self.sample_episode_num = sample_episode_num
        self.evaluate_episode_num = evaluate_episode_num
        self.pretrain_dir = pretrain_dir

        mode = mode.lower()
        if mode == "eval":
            # Read some basic env and model info from the dir configs
            assert load_dir is not None, "The load_path parameter has not been specified!!!"
            model_path, env, policy, timeout_steps, policy_config = setup_eval_configs(
                load_dir)
            self._eval_mode_init(env, seed, model_path, policy, timeout_steps,
                                 policy_config, **kwarg)
        else:
            self._train_mode_init(env, seed, exp_name, policy, timeout_steps, data_dir,
                                  **kwarg)
            self.batch_size = self.worker_config[
                "batch_size"] if "batch_size" in self.worker_config else None

        self.epochs = epochs
        self.save_freq = save_freq
        self.data_dict = []
        self.epoch = 0
        self.verbose = verbose
        self.env_name = env
        if mode == "train" and "cost_limit" in self.policy_config:
            self.cost_limit = self.policy_config["cost_limit"]
        else:
            self.cost_limit = 1e3

    def _train_mode_init(self, env, seed, exp_name, policy, timeout_steps, data_dir,
                         **kwarg):

        # record some local attributes from the child classes
        attrs = deepcopy(self.__dict__)

        # Instantiate environment
        # env_config = {
        #     "_seed": seed,
        #     "num_different_layouts": kwarg["env_layout_nums"],
        # }
        # self.env = gym.make(env, config=env_config)
        if env == "waymo":
            self.env = AddCostToRewardEnv(kwarg["waymo_config"])
        else:

            self.env = gym.make(env)
            self.env.seed(kwarg["env_seed"])
        if "Safexp" in env:
            self.env.set_num_different_layouts(kwarg["env_layout_nums"])
        self.timeout_steps = self.env._max_episode_steps if timeout_steps == -1 else timeout_steps

        # Set up logger and save configuration
        logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=data_dir)
        self.logger = EpochLogger(**logger_kwargs)

        config = locals()
        config.update(attrs)

        # remove some non-useful keys
        [config.pop(key) for key in ["self", "logger_kwargs", "kwarg", "attrs"]]
        config[policy] = kwarg[policy]
        self.logger.save_config(config)

        # Init policy
        self.policy_config = kwarg[policy]
        self.policy_config["timeout_steps"] = self.timeout_steps
        policy_cls, self.on_policy, worker_cls = self.POLICY_LIB[policy.lower()]
        self.policy = policy_cls(self.env, self.logger, **self.policy_config)

        if self.pretrain_dir is not None:
            model_path, _, _, _, _ = setup_eval_configs(self.pretrain_dir)
            self.policy.load_model(model_path)

        self.steps_per_epoch = self.policy_config[
            "steps_per_epoch"] if "steps_per_epoch" in self.policy_config else 1
        self.worker_config = self.policy_config["worker_config"]

        if "jp" in policy.lower():
            self.worker_config["expert_policies"] = {} 
            dummy_logger = EpochLogger(output_dir="data/test", use_tensor_board=False)
            if isinstance(self.worker_config["model_dir"], dict):
                model_dir = self.worker_config["model_dir"][self.env.get_seed()]
            else:
                model_dir = self.worker_config["model_dir"]
            if self.worker_config["use_dt_guide"]:
                loaded_stats = js_utils.load_demo_stats(
                    path=model_dir
                )
                obs_mean, obs_std, reward_scale, target_return_init = loaded_stats
                self.worker_config["obs_mean"] = obs_mean
                self.worker_config["obs_std"] = obs_std
                self.worker_config["reward_scale"] = reward_scale
                self.worker_config["target_return_init"] = target_return_init
                expert = js_utils.load_transformer(
                    model_dir=model_dir, device=self.worker_config["device"]
                )
            else:
                expert = BC(self.env, dummy_logger)
                expert.load_model(model_dir)
            self.worker_config["expert_policies"][self.env.get_seed()] = expert 

            if self.worker_config["load_critic"]:
                self.policy.load_critic(model_dir)
                print("Successfully Loaded Critic!\n")
        if self.worker_config.get("load_actor", False):
            model_dir = self.worker_config["model_dir"][self.env.get_seed()]
            self.policy.load_actor(model_dir)
        self.add_bc_loss = self.worker_config.get("add_bc_loss", False)
            
        self.worker = worker_cls(self.env,
                                 self.policy,
                                 self.logger,
                                 timeout_steps=self.timeout_steps,
                                 **self.worker_config)
        
        expert_data_dir = self.worker_config.get("expert_data_dir", None)
        if self.add_bc_loss:
            self.worker.load_expert_cpp_buffer(expert_data_dir)
        if "bc" == policy.lower():
            self.worker.load_cpp_buffer(expert_data_dir)


    def _eval_mode_init(self, env, seed, model_path, policy, timeout_steps,
                        policy_config,**kwarg):
        # Instantiate environment
        if env == 'waymo':
            waymo_test_config = kwarg["waymo_config"].copy()
            waymo_test_config['start_seed'] = 10000
            waymo_test_config['case_num'] = 100
            self.env = AddCostToRewardEnv(waymo_test_config)
            self.eval_fn =  "js-cvpo" if kwarg['use_dt_guide'] else "cvpo"
            self.save_fig_dir =  kwarg['save_fig_dir'] 
        else:
            self.env = gym.make(env)
        # self.env.seed(seed)
        self.timeout_steps = self.env._max_episode_steps if timeout_steps == -1 else timeout_steps

        # Set up logger but don't save anything
        self.logger = EpochLogger(eval_mode=True)

        # Init policy
        policy_config["timeout_steps"] = self.timeout_steps

        policy_cls, self.on_policy, worker_cls = self.POLICY_LIB[policy.lower()]
        self.policy = policy_cls(self.env, self.logger, **policy_config)

        self.policy.load_model(model_path)

    # @profile
    def train_one_epoch_off_policy(self, epoch):
        epoch_steps = 0
        range_instance = tqdm(
            range(self.sample_episode_num),
            desc='Collecting trajectories') if self.verbose else range(
                self.sample_episode_num)
        for i in range_instance:
            steps = self.worker.work()
            epoch_steps += steps
        
        if self.sample_episode_num > 0:
            train_steps = self.episode_rerun_num * epoch_steps // self.batch_size
        else:
            train_steps = 1000
            epoch_steps = 8000
        range_instance = tqdm(
            range(train_steps), desc='training {}/{}'.format(
                epoch + 1, self.epochs)) if self.verbose else range(train_steps)
        for i in range_instance:
            if self.add_bc_loss:
                expert_data = self.worker.get_expert_sample()
                self.policy.learn_on_expert_batch(expert_data)
            data = self.worker.get_sample()
            self.policy.learn_on_batch(data)
        return epoch_steps

    # @profile
    def train_one_epoch_on_policy(self, epoch):
        epoch_steps = 0
        steps = self.worker.work()
        epoch_steps += steps
        data = self.worker.get_sample()
        self.policy.learn_on_batch(data)
        return epoch_steps

    def train(self):
        start_time = time.time()
        total_steps = 0
        for epoch in range(self.epochs):
            self.epoch += 1
            if self.on_policy:
                epoch_steps = self.train_one_epoch_on_policy(epoch)
            else:
                epoch_steps = self.train_one_epoch_off_policy(epoch)
            total_steps += epoch_steps

            for _ in range(self.evaluate_episode_num):
                self.worker.eval()

            if hasattr(self.policy, "post_epoch_process"):
                self.policy.post_epoch_process()
            if hasattr(self.worker, "post_epoch_process"):
                self.worker.post_epoch_process(epoch)

            # Save model
            # if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
            #     self.logger.save_state({'env': self.env}, None)
            # Log info about epoch
            self.data_dict = self._log_metrics(epoch, total_steps,
                                               time.time() - start_time, self.verbose)

    def eval(self, epochs=10, sleep=0.01, render=True):
        if "Safety" in self.env_name:
            render = False
            self.env.render()
        total_steps = 0

        start_id = int(self.env.config['start_case_index'])
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        episode_success_rates = []


        for epoch in range(epochs):
            
            obs = self.env.reset(force_seed = start_id + epoch)
            ep_cost, current_reward, current_length = 0, 0, 0
            if render:
                self.env.render()
            for i in range(self.timeout_steps):
                res = self.policy.act(obs, deterministic=True, with_logprob=False)
                action = res[0]
                obs_next, reward, done, info = self.env.step(action)
                current_reward += reward
                current_length += 1
                if render:
                    self.env.render()
                time.sleep(sleep)

                if "cost" in info:
                    ep_cost += info["cost"]

                total_steps += 1
                obs = obs_next

                if done:
                    episode_rewards.append(current_reward)
                    episode_costs.append(ep_cost)
                    episode_lengths.append(current_length)
                    episode_success_rates.append(info['arrive_dest'])
                    break
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_cost = np.mean(episode_costs)
        std_cost = np.std(episode_costs)
        mean_success_rate = np.mean(episode_success_rates)

        fn =  self.eval_fn
        header = "-"*10+" Evaluation of " + fn + "-"*10

            
        print(header)
        print("mean_reward = ", mean_reward)
        print("std_reward = ",std_reward)
        print("mean_cost = ", mean_cost)
        print("std_cost = ",std_cost)
        print("mean_success_rate = ", mean_success_rate)

        exp_result ={
            "exp_name": fn,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_cost": mean_cost,
            "std_cost": std_cost,
            "mean_success_rate": mean_success_rate
        }
        json_path = fn + ".json"

        # Save the dictionary as a JSON config file
        with open(json_path, "w") as json_file:
            json.dump(exp_result, json_file, indent=4)      
        
        for seed in tqdm.tqdm(range(  start_id,   start_id + epoch)):
            plot_waymo_vs_pred(self.env, 
                            self.policy, 
                            seed, 
                            fn, 
                            savefig_dir = os.path.join(self.save_fig_dir, fn), 
                            end_eps_when_done = True)

        


    def _log_metrics(self, epoch, total_steps, time=None, verbose=True):
        self.logger.log_tabular('CostLimit', self.cost_limit)
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('TotalEnvInteracts', total_steps)
        for key in self.logger.logger_keys:
            self.logger.log_tabular(key, average_only=True)
        if time is not None:
            self.logger.log_tabular('Time', time)
        # data_dict contains all the keys except Epoch and TotalEnvInteracts
        data_dict = self.logger.dump_tabular(
            x_axis="TotalEnvInteracts",
            verbose=verbose,
            env=self.env_name,
        )
        return data_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Pendulum-v0')
    parser.add_argument('--policy', '-p', type=str, default='ppo')
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--load_dir', '-d', type=str, default='None')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default="gpu")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='None')
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    args = parser.parse_args()
    if args.exp_name == 'None':
        # default name of the experiments
        args.exp_name = args.env.split("-")[0] + '_' + args.policy
    args_dict = vars(args)

    config = load_config("safe_rl/config/default_config.yaml")
    config.update(args_dict)

    runner = Runner(**config)
    if args.mode == "train":
        runner.train()
    else:
        runner.eval(render=not args.no_render, sleep=args.sleep)
