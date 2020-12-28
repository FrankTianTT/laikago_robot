from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from builder.env_builder import build_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")
    parser.add_argument("-lv", "--load_version")
    parser.add_argument('-t', '--task_name', default='standup')
    parser.add_argument('-m', '--run_mode', default='train')
    parser.add_argument('-s', '--simulator', default='mujoco')

    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--time_steps", default=5000000)
    parser.add_argument("--buffer_size", default=1000000)
    parser.add_argument("--learning_starts", default=1000)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--ent_coef", default='auto')
    parser.add_argument("--net_arch", default=[256, 256], nargs='+', type=int)

    args = parser.parse_args()

    learning_rate = args.learning_rate
    version = args.version
    task_name = args.task_name
    run_mode = args.run_mode
    simulator = args.simulator

    buffer_size = args.buffer_size
    batch_size = args.batch_size
    learning_starts = args.learning_starts
    ent_coef = args.ent_coef
    time_steps = args.time_steps
    net_arch = args.net_arch

    log_path = './{}/{}/SAC-v{}/logs/'.format(simulator, task_name, version)
    tensorboard_log = './{}/{}/SAC-v{}/log/'.format(simulator, task_name, version)
    best_model_save_path = './{}/{}/SAC-v{}/logs/'.format(simulator, task_name, version)

    env = build_env(task_name, version, run_mode, simulator, visual=False, ctrl_delay=True)
    eval_env = build_env(task_name, version, run_mode, simulator, visual=False, ctrl_delay=True)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_save_path,
                                 log_path=log_path,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)

    if args.load_version is not None:
        best_model_dir = './{}/{}/SAC-v{}/logs/best_model.zip'.format(simulator, task_name, args.load_version)
        model = SAC.load(best_model_dir, device=torch.device('cuda:0'))
        model.set_env(env)
        model.tensorboard_log = tensorboard_log
        model.num_timesteps = 0
        model.learning_starts = args.learning_starts
        model.buffer_size = args.buffer_size
        model.learning_rate = learning_rate
        if ent_coef == 'auto':
            init_value = 1.0
            model.log_ent_coef = torch.log(torch.ones(1, device=model.device) * init_value).requires_grad_(True)
            model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))

    else:
        model = SAC('MlpPolicy',
                    env,
                    gamma=0.99,
                    learning_rate=learning_rate,
                    device=torch.device('cuda:0'),
                    verbose=1,
                    tensorboard_log=tensorboard_log,
                    policy_kwargs=policy_kwargs,
                    buffer_size=buffer_size,
                    batch_size=batch_size,
                    learning_starts=learning_starts,
                    ent_coef=ent_coef)
    model.learn(total_timesteps=time_steps, callback=eval_callback)