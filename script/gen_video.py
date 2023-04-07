from safe_rl.policy import CVPO, SAC 
from safe_rl.util.logger import EpochLogger
import safety_gym
import gym
import numpy as np
import torch
import tqdm
from safe_rl.util.torch_util import export_device_env_variable, seed_torch


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)

seed = 0
device = 'cpu'
device_id = 0
threads = 2
seed_torch(seed)
torch.set_num_threads(threads)
export_device_env_variable(device, id=device_id)

env_name = 'Safexp-CarGoal1-v0'
env = gym.make(env_name)

logger = EpochLogger(output_dir="data/test", use_tensor_board=False)
policy = SAC(env, logger)
policy.load_model('data/Safexp-CarGoal1-v0_cost_10/sac_epoch_150_random-start/sac_epoch_150_random-start_s0/model_save/model.pt')

total_timesteps = 2000
imgs = []
obs = env.reset()
for i in tqdm.tqdm(range(total_timesteps)):
    action, _ = policy.act(obs, deterministic=True, with_logprob=False)
    obs_next, reward, done, info = env.step(action)
    img = env.sim.render(600, 600)
    # img = env.render(mode='rgb_array', camera_id=1)
    imgs.append(img)
    if done:
        obs = env.reset()
    else:
        obs = obs_next 
env.close()

imgs = np.array(imgs)
# np.save('data/' + env_name, imgs)

frames = imgs
# frames = frames.transpose((0, 3, 1, 2))
# import pdb;pdb.set_trace()
# frames = unnormalize_image(frames)

# save it as a gif
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip(list(frames), fps=160)
clip.write_gif('data/videos/goal1-sac-random-start.gif', fps=20)
