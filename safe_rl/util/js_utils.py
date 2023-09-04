import json
import os
import re

import numpy as np
import torch
import yaml

from safe_rl.policy.model.decision_transformers import DecisionTransformer

    
def load_demo_stats(path):
    stats_path = os.path.join(path, 'obs_stats.json')
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return (
        np.array(stats['obs_mean']), 
        np.array(stats['obs_std']),
        stats['reward_scale'],
        stats['target_return'],
    )

def load_transformer(model_dir, device):
    if model_dir.split('/')[-1] == 'model.pt':
        raise ValueError("Please use the root dir of model.pt!")
    config_path = os.path.join(model_dir, 'config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    model = DecisionTransformer(
        state_dim=config['state_dim']['value'],
        act_dim=config['act_dim']['value'],
        max_length=config['K']['value'],
        max_ep_len=config['max_ep_len']['value'],
        hidden_size=config['embed_dim']['value'], # default 128
        n_layer=config['n_layer']['value'],
        n_head=config['n_head']['value'],
        n_inner=4*config['embed_dim']['value'],
        activation_function=config['activation_function']['value'],
        n_positions=1024,
        resid_pdrop=config['dropout']['value'],
        attn_pdrop=config['dropout']['value'],
    )
    state_dict_path = os.path.join(model_dir, 'model.pt')
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()
    return model