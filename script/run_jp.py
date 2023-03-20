import os.path as osp

from safe_rl.runner import Runner
from safe_rl.util.run_util import load_config

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")


class SafetyGymRunner(Runner):

    def eval(self, epochs=10, sleep=0.01, render=True):
        '''
        Overwrite the eval function since the rendering for 
        bullet safety gym is different from other gym envs
        '''
        if render:
            self.env.render()
        super().eval(epochs, sleep, False)


EXP_NAME_KEYS = {"epochs": "epoch"}
DATA_DIR_KEYS = {"cost_limit": "cost"}


def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name + suffix


def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-PointButton1-v0')
    parser.add_argument('--policy', '-p', type=str, default='cvpo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)
    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    args = parser.parse_args()

    args_dict = vars(args)

    if 'cvpo' in args.policy:
        config_path = osp.join(CONFIG_DIR, "config_cvpo.yaml")
    else:
        config_path = osp.join(CONFIG_DIR, "config_baseline.yaml")
    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    model_dirs = {
        "Safexp-CarGoal1-v0": 
            "data/Safexp-CarGoal1-v0_cost_10/sac_epoch_300_new_env/" + 
            "sac_epoch_300_new_env_s0/model_save/model.pt",
        "Safexp-CarButton1-v0":
            "data/Safexp-CarButton1-v0_cost_10/sac_epoch_300_new_env/" +
            "sac_epoch_300_new_env_s0/model_save/model.pt",
        "Safexp-CarPush1-v0":
            "data/Safexp-CarPush1-v0_cost_10/sac_epoch_300_new_env/" +
            "sac_epoch_300_new_env_s0/model_save/model.pt",
    }
    assert args.env in model_dirs, f"No pretrained model for {args.env}!"
    config[args.policy]["worker_config"]["model_dir"] = model_dirs[args.env]

    if "Safety" in args.env:
        runner = SafetyGymRunner(**config)
    else:
        runner = Runner(**config)
    
    if args.mode == "train":
        runner.train()
    else:
        runner.eval(render=not args.no_render, sleep=args.sleep)
