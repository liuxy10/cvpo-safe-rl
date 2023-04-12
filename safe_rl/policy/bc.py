from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safe_rl.policy.base_policy import Policy
from safe_rl.policy.model.mlp_ac import CholeskyGaussianActor, EnsembleQCritic
from safe_rl.util.logger import EpochLogger
from safe_rl.util.torch_util import (count_vars, get_device_name, to_device,
                                     to_ndarray, to_tensor)
from torch.distributions import MultivariateNormal
from torch.optim import Adam


class BC(Policy):
    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 actor_lr=0.001,
                 critic_lr=0.001,
                 ac_model="mlp",
                 hidden_sizes=[64, 64],
                 alpha=0.2,
                 gamma=0.99,
                 polyak=0.995,
                 num_q=2,
                 **kwargs) -> None:
        r'''
        Behavior Cloning (BC)

        Args:
        @param env : The environment must satisfy the OpenAI Gym API.
        @param logger: Log useful informations, and help to save model
        @param actor_lr, critic_lr (float): Learning rate for policy and Q-value learning.
        @param ac_model: the actor critic model name
        @param alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)
        @param gamma (float): Discount factor. (Always between 0 and 1.)
        @param polyak (float): Interpolation factor in polyak averaging for target 
        @param num_q (int): number of models in the q-ensemble critic.
        '''
        super().__init__()

        self.logger = logger
        self.actor_lr = actor_lr
        self.hidden_sizes = hidden_sizes

        ################ create actor critic model ###############
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_lim = env.action_space.high[0]
        '''
        Notice: The output action are normalized in the range [-1, 1], 
        so please make sure your action space's high and low are suitable
        '''

        if ac_model.lower() == "mlp":
            if isinstance(env.action_space, gym.spaces.Box):
                # Action limit for normalization: critically, assumes all dimensions share the same bound!
                self.act_lim = env.action_space.high[0]
                actor = CholeskyGaussianActor(self.obs_dim, self.act_dim, -self.act_lim,
                                            self.act_lim, hidden_sizes, nn.ReLU)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                raise ValueError("Discrete action space does not support yet")
        else:
            raise ValueError(f"{ac_model} ac model does not support.")

        # Set up optimizer and target q models
        self._ac_training_setup(actor)

        # Set up model saving
        self.save_model()

        # Count variables
        var_counts = count_vars(self.actor)
        self.logger.log(
            '\nNumber of parameters: \t actor pi: %d \n' %
            var_counts)

    def _ac_training_setup(self, actor):
        self.actor = to_device(actor, get_device_name())

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, logp.
        This API is used to interact with the env.

        @param obs (1d ndarray): observation
        @param deterministic (bool): True for evaluation mode, which returns the action with highest pdf (mean).
        @param with_logprob (bool): True to return log probability of the sampled action, False to return None
        @return act, logp, (1d ndarray)
        '''
        obs = to_tensor(obs).reshape(1, -1)
        logp_a = None
        with torch.no_grad():
            mean, cholesky, pi_dist = self.actor_forward(obs)
            a = mean if deterministic else pi_dist.sample()
            logp_a = pi_dist.log_prob(a) if with_logprob else None
        # squeeze them to the right shape
        a, logp_a = np.squeeze(to_ndarray(a), axis=0), np.squeeze(to_ndarray(logp_a))
        return a, logp_a

    def learn_on_batch(self, data: dict):
        '''
        Given a batch of data, train the policy
        data keys: (obs, act, rew, obs_next, done)
        '''
        self._update_actor(data)


    def actor_forward(self, obs, return_pi=True):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, (tensor), [batch, obs_dim]
        @return mean, (tensor), [batch, act_dim]
        @return cholesky, (tensor), (batch, act_dim, act_dim)
        @return pi_dist, (MultivariateNormal)
        '''
        mean, cholesky = self.actor(obs)
        pi_dist = MultivariateNormal(mean, scale_tril=cholesky) if return_pi else None
        return mean, cholesky, pi_dist

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        def policy_loss():
            obs = data['obs']
            target = data['act']
            pred, _ = self.actor.forward(obs)

            # Entropy-regularized policy loss
            assert pred.shape == target.shape
            loss_pi = F.mse_loss(pred, target)

            # Useful info for logging

            return loss_pi

        self.actor_optimizer.zero_grad()
        loss_pi = policy_loss()
        loss_pi.backward()
        self.actor_optimizer.step()

        # Log actor update info
        self.logger.store(LossPi=loss_pi.item())


    def save_model(self):
        self.logger.setup_pytorch_saver((self.actor))

    def load_model(self, path):
        actor = torch.load(path)
        self._ac_training_setup(actor)
