from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from builder.gym_env import LaikagoEnv
import builder.tasks_sim as tasks_sim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")
    parser.add_argument("-l", "--load_from_best", default=False, type=bool)

    parser.add_argument("--time_steps", default=5000000)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--net_arch", default=[256, 256], nargs='+', type=int)

    args = parser.parse_args()

    version = args.version
    batch_size = args.batch_size
    time_steps = args.time_steps
    net_arch = args.net_arch

    log_path = './PPO-v{}/logs/'.format(version)
    tensorboard_log = './PPO-v{}/log/'.format(version)
    best_model_save_path = './PPO-v{}/logs/'.format(version)
    best_model_dir = './PPO-v{}/logs/best_model.zip'.format(version)

    standup_task_sim = importlib.import_module('builder.tasks_sim.standup_task_sim')
    task = eval('standup_task_sim.LaikagoStandUpSim{}()'.format(version))

    env = LaikagoEnv(task=task, visual=False)
    eval_env = LaikagoEnv(task=task, visual=False)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_save_path,
                                 log_path=log_path,
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=net_arch)

    model = PPO('MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                batch_size=batch_size)

    if args.load_from_best:
        model = PPO.load(best_model_dir)
        model.set_env(env)
    model.learn(total_timesteps=time_steps, callback=eval_callback)