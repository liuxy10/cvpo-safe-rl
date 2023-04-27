import gym
import numpy as np
import torch
from cpprb import ReplayBuffer
from safe_rl.policy.base_policy import Policy
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import to_tensor, to_ndarray

class JumpStartOffPolicyWorker:
    r'''
    Collect data based on the policy and env, and store the interaction data to data buffer.
    '''
    def __init__(self,
                 env: gym.Env,
                 policy: Policy,
                 logger: EpochLogger,
                 batch_size=100,
                 timeout_steps=200,
                 buffer_size=1e6,
                 warmup_steps=10000,
                 **kwargs) -> None:
        self.env = env
        self.policy = policy
        self.expert = kwargs["expert_policies"]
        self.logger = logger
        self.batch_size = batch_size
        self.timeout_steps = timeout_steps

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
            self.SAFE_RL_ENV = True
            env_dict["cost"] = {'dtype': np.float32}
        self.cpp_buffer = ReplayBuffer(buffer_size, env_dict)
        self.expert_cpp_buffer = ReplayBuffer(buffer_size, env_dict)
        self.eval_max_rew = -float("inf")

        self.last_obs_reset = None

        # ######### Warmup phase to collect data with random policy #########
        # steps = 0
        # while steps < warmup_steps:
        #     steps += self.work(warmup=True)

        # ######### Train the policy with warmup samples #########
        # for i in range(warmup_steps // 2):
        #     self.policy.learn_on_batch(self.get_sample())

    def reset_guidance_steps(self, upper_bound):
        self.guidance_steps = np.random.randint(1, upper_bound-1)

    def work(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        idx = self.env.get_seed()
        if self.last_obs_reset is not None and self.env.num_different_layouts == 1:
            assert np.sum(obs - self.last_obs_reset) < 1e-6
        self.last_obs_reset = obs
        self.reset_guidance_steps(self.timeout_steps)

        epoch_steps = 0
        terminal_freq = 0
        done_freq = 0
        last_steps = -1
        q_expert, q_train = [], []
        for i in range(self.timeout_steps):
            # TODO: Change here for jump start
            if i-last_steps <= self.guidance_steps:
                action, _ = self.expert[idx].act(obs, 
                                            deterministic=False,
                                            with_logprob=False)
                with torch.no_grad():
                    _, q_list = self.policy.critic.predict(to_tensor(obs), to_tensor(action))
                q_expert += q_list
                action_policy, _ = self.policy.act(obs,
                                            deterministic=False,
                                            with_logprob=False)
                with torch.no_grad():
                    _, q_list = self.policy.critic.predict(to_tensor(obs), to_tensor(action_policy))
                q_train += q_list
            else:
                action, _ = self.policy.act(obs,
                                            deterministic=False,
                                            with_logprob=False)
                
            obs_next, reward, done, info = self.env.step(action)
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if i == self.timeout_steps - 1 or "TimeLimit.truncated" in info else done
            # ignore the goal met terminal condition
            terminal = done
            done = True if "goal_met" in info and info["goal_met"] else done

            if done:
                done_freq += 1
                # self.reset_guidance_steps(i - last_steps + 1)
                # last_steps = i


            if "cost" in info:
                cost = info["cost"]
                ep_cost += cost
                self.cpp_buffer.add(obs=obs,
                                    act=np.squeeze(action),
                                    rew=reward,
                                    obs2=obs_next,
                                    done=done,
                                    cost=cost)
            else:
                self.cpp_buffer.add(obs=obs,
                                    act=np.squeeze(action),
                                    rew=reward,
                                    obs2=obs_next,
                                    done=done)
            ep_reward += reward
            ep_len += 1
            epoch_steps += 1
            obs = obs_next
            if terminal:
                terminal_freq += 1
                self.logger.store(EpRet=ep_reward,
                                  EpCost=ep_cost,
                                  EpLen=ep_len,
                                  tab="worker")
                obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
                idx = self.env.get_seed()
                self.reset_guidance_steps(self.timeout_steps)
                # break
        q_expert = to_ndarray(torch.hstack(q_expert)) if q_expert else 0
        q_train = to_ndarray(torch.hstack(q_train)) if q_train else 0
        self.logger.store(EpRet=ep_reward,
                          EpCost=ep_cost,
                          EpLen=ep_len,
                          Terminal=terminal_freq,
                          Done=done_freq,
                          Q_Expert=q_expert,
                          Q_Train=q_train,
                          Q_ExpertMinusTrain=q_expert.mean()-q_train.mean(),
                          GuideFraction=self.guidance_steps/ep_len,
                          tab="worker")
        return epoch_steps

    def eval(self):
        '''
        Interact with the environment to collect data
        '''
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        for i in range(self.timeout_steps):
            action, _ = self.policy.act(obs, deterministic=True, with_logprob=False)
            obs_next, reward, done, info = self.env.step(action)
            if "cost" in info:
                cost = info["cost"]
                ep_cost += cost
            ep_reward += reward
            ep_len += 1
            obs = obs_next
            if done:
                break

        if ep_reward > self.eval_max_rew:
            self.eval_max_rew = ep_reward
            self.logger.save_state({'env': self.env}, None)

        self.logger.store(TestEpRet=ep_reward,
                          TestEpLen=ep_len,
                          TestEpCost=ep_cost,
                          tab="eval")

    def get_sample(self):
        data = to_tensor(self.cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data
    
    def get_expert_sample(self):
        data = to_tensor(self.expert_cpp_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data
    
    def load_expert_cpp_buffer(self, expert_data_dir):
        self.expert_cpp_buffer.load_transitions(expert_data_dir)

    def clear_buffer(self):
        self.cpp_buffer.clear()
