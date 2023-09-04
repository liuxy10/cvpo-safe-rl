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
                 use_jp_decay=False,
                 decay_epoch=100,
                 **kwargs) -> None:
        self.env = env
        self.policy = policy
        self.expert = kwargs["expert_policies"]
        self.use_dt_guide = kwargs["use_dt_guide"]
        if self.use_dt_guide:
            self.obs_mean = kwargs["obs_mean"]
            self.obs_std = kwargs["obs_std"]
            self.reward_scale = kwargs["reward_scale"]
            self.target_return_init = kwargs["target_return_init"]
            self.device = kwargs["device"]
        self.guidance_timesteps = kwargs["guidance_timesteps"]
        self.logger = logger
        self.batch_size = batch_size
        self.timeout_steps = timeout_steps
        self.num_timesteps = 0  # Total num of env steps

        self.use_jp_decay = use_jp_decay
        self.decay_epoch = decay_epoch

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape

        env_dict = {
            'act': {
                'dtype': np.float32,
                'shape': self.act_dim
            },
            'done': {
                'dtype': np.float32,
            },
            'obs': {
                'dtype': np.float32,
                'shape': self.obs_dim
            },
            'obs2': {
                'dtype': np.float32,
                'shape': self.obs_dim
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

    def get_guide_probability(self):
        if self.num_timesteps > self.guidance_timesteps:
            return 0.
        prob_start = 0.9
        prob = prob_start * np.exp(-5. * self.num_timesteps / self.guidance_timesteps)
        return prob

    def work(self, warmup=False):
        '''
        Interact with the environment to collect data
        '''
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        idx = self.env.get_seed()
        if self.last_obs_reset is not None and self.env.num_different_layouts == 1:
            assert np.sum(obs - self.last_obs_reset) < 1e-6
        self.last_obs_reset = obs
        self.hist_obs = obs.reshape(1, self.obs_dim)
        self.hist_ac= np.zeros((0, self.act_dim[0]))
        self.hist_re = np.zeros(0)
        if self.use_dt_guide:
            self.target_return = np.array([[self.target_return_init]])
        self.timesteps = np.zeros((1, 1))

        epoch_steps = 0
        terminal_freq = 0
        done_freq = 0
        last_steps = -1
        q_expert, q_train = [], []
        for i in range(self.timeout_steps):
            self.hist_ac = np.concatenate(
                [self.hist_ac, np.zeros((1, self.act_dim[0]))], axis=0
            )
            self.hist_re = np.concatenate([self.hist_re, np.zeros(1)])
            guide_prob = self.get_guide_probability()
            use_guide = np.random.choice([False, True], p=[1-guide_prob, guide_prob])
            if use_guide:
                if self.use_dt_guide:
                    hist_obs = torch.tensor(
                        self.hist_obs, dtype=torch.float32, device=self.device
                    )
                    hist_ac = torch.tensor(
                        self.hist_ac, dtype=torch.float32, device=self.device
                    )
                    hist_re = torch.tensor(
                        self.hist_re, dtype=torch.float32, device=self.device
                    )
                    target_return = torch.tensor(
                        self.target_return, dtype=torch.float32, device=self.device
                    )
                    timesteps = torch.tensor(
                        self.timesteps, dtype=torch.long, device=self.device
                    )
                    action = self.expert[idx].get_action(
                        (hist_obs - self.obs_mean) / self.obs_std,
                        hist_ac,
                        hist_re,
                        target_return,
                        timesteps,
                    )
                    action = action.detach().cpu().numpy()
                    self.hist_ac[-1] = action
                else:
                    action, _ = self.expert[idx].act(
                        obs, 
                        deterministic=False,
                        with_logprob=False
                    )
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
            self.num_timesteps += 1

            self.hist_obs = np.concatenate(
                [self.hist_obs, obs_next.reshape(1, -1)], axis=0)
            assert len(self.hist_obs.shape) == 2
            self.hist_re[-1] = reward
            pred_return = self.target_return[0,-1] - (reward/self.reward_scale)
            self.target_return = np.concatenate(
                [self.target_return, pred_return.reshape(1, 1)], axis=1)
            t = self.timesteps[0, -1] + 1
            self.timesteps = np.concatenate(
                [self.timesteps, np.ones((1, 1)) * t], axis=1)
            
            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if i == self.timeout_steps - 1 or "TimeLimit.truncated" in info else done
            # ignore the goal met terminal condition
            terminal = done
            done = True if "goal_met" in info and info["goal_met"] else done

            if done:
                done_freq += 1


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
                
                self.hist_obs = obs.reshape(1, self.obs_dim)
                self.hist_ac= np.zeros((0, self.act_dim[0]))
                self.hist_re = np.zeros(0)
                if self.use_dt_guide:
                    self.target_return = np.array([[self.target_return_init]])
                self.timesteps = np.zeros((1, 1))

                idx = self.env.get_seed()
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
