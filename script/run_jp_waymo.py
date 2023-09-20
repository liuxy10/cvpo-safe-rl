import os.path as osp

from safe_rl.runner import Runner
from safe_rl.util.run_util import load_config
from metadrive.policy.replay_policy import PMKinematicsEgoPolicy
from tqdm import tqdm


CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")



EXP_NAME_KEYS = {"epochs": "epoch", "env_layout_nums": "layouts", "env_seed": "es"}
DATA_DIR_KEYS = {"cost_limit": "cost"}


def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    if config['use_dt_guide']:
        name += '_use_dt_guide'
    return name + suffix


def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='waymo')
    parser.add_argument('--env_layout_nums', '-eln', type=int, default=1)
    parser.add_argument('--policy', '-p', type=str, default='cvpo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)
    parser.add_argument('--guidance_timesteps', '-gt', type=int, default=int(1e6))
    parser.add_argument('--use_dt_guide', '-dt', action="store_true")
    parser.add_argument('--model_dir', '-md', type=str, default=None)
    parser.add_argument('--load_dir', '-d', type=str, default="data/waymo_cost_10/cvpo_epoch_2500_layouts_1_es_0/cvpo_epoch_2500_layouts_1_es_0_s0")
    parser.add_argument('--mode', '-m', type=str, default='eval')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--env_seed', '-es', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    parser.add_argument('--load_critic', action="store_true")
    parser.add_argument('--load_actor', action="store_true")
    parser.add_argument('--bc_loss', action="store_true")
    parser.add_argument('--save_fig_dir', type=str, default="./figs/")

    parser.add_argument('--lamb', '-lm', type=float, default=1.)
    parser.add_argument('--num_of_scenarios', type=int, default=10)
    parser.add_argument('--pkl_dir', '-pkl', type=str,
                        default='/home/xinyi/src/data/metadrive/pkl_9')
    
    args = parser.parse_args()

    args_dict = vars(args)

    if 'cvpo' in args.policy:
        config_path = osp.join(CONFIG_DIR, "config_cvpo.yaml")
    elif args.policy == 'bc':
        config_path = osp.join(CONFIG_DIR, "config_bc.yaml")
    else:
        config_path = osp.join(CONFIG_DIR, "config_baseline.yaml")
    config = load_config(config_path)
    config.update(args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    model_dirs = {
        "waymo":
            '/home/xinyi/src/decision-transformer/gym/wandb/run-20230823_230743-3s6y7mzy'    }
    # assert args.env in model_dirs, f"No pretrained model for {args.env}!"
    config[args.policy]["worker_config"]["use_jp_decay"] = config[args.policy]["use_jp_decay"]
    config[args.policy]["worker_config"]["decay_epoch"] = config[args.policy]["decay_epoch"]
    config[args.policy]["worker_config"]["model_dir"] = (
        model_dirs[args.env] 
        if config["model_dir"] is None 
        else config["model_dir"]
    )
    config[args.policy]["worker_config"]["use_dt_guide"] = config["use_dt_guide"]
    config[args.policy]["worker_config"]["device"] = config["device"]
    config[args.policy]["worker_config"]["guidance_timesteps"] = (
        config["guidance_timesteps"]
    )

    config[args.policy]["worker_config"]["load_critic"] = config["load_critic"]
    config[args.policy]["worker_config"]["load_actor"] = config["load_actor"]
    config[args.policy]["worker_config"]["add_bc_loss"] = (
        config["bc_loss"])
    config[args.policy]["worker_config"]["expert_data_dir"] = (
        "data/expert_data_" + args.env + "_s2.npz"
    )
    if args.env == "waymo":
        config['waymo_config'] = {
            "manual_control": False,
            "no_traffic": False,
            "agent_policy": PMKinematicsEgoPolicy,
            "start_seed": 0,
            "waymo_data_directory": args.pkl_dir,
            "case_num": args.num_of_scenarios,
            # have to be specified each time we use waymo environment for training purpose
            "physics_world_step_size": 1/10,
            "use_render": False,
            "horizon": 90/5,
            "reactive_traffic": False,
            "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=80, distance=50, num_others=4),  # 120
                lane_line_detector=dict(num_lasers=12, distance=50),  # 12
                side_detector=dict(num_lasers=20, distance=50))  # 160,
        }

    
    runner = Runner(**config)
    
    if args.mode == "train":
        runner.train()
    else:
        # runner.eval(render=not args.no_render, sleep=args.sleep)
        runner.eval(render=False, sleep=args.sleep, epochs= 100)
