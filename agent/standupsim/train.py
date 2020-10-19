from stable_baselines3 import SAC
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
    parser.add_argument("--buffer_size", default=1000000)
    parser.add_argument("--learning_starts", default=10000)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--ent_coef", default='auto')
    parser.add_argument("--net_arch", default=[256, 256], nargs='+', type=int)

    args = parser.parse_args()

    version = args.version
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    learning_starts = args.learning_starts
    ent_coef = args.ent_coef
    time_steps = args.time_steps
    net_arch = args.net_arch

    log_path = './v{}/logs/'.format(version)
    tensorboard_log = './v{}/log/'.format(version)
    best_model_save_path = './v{}/logs/'.format(version)

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

    model = SAC('MlpPolicy',
                env,
                verbose=1,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                ent_coef=ent_coef)

    if args.load_from_best:
        model = SAC.load(best_model_save_path)
        model.set_env(env)
    model.learn(total_timesteps=time_steps, callback=eval_callback)