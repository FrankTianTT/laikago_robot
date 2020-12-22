from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, dirname(dirname(dirname(dirname(abspath(__file__))))))
from builder.env_builder import build_env
import numpy as np

TASK_NAME = 'standup'
ClASS_NAME = 'StandUp'
RUN_MODE = 'report_done'
SIMULATOR = 'bullet_torque'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")
    parser.add_argument("-lv", "--load_version")

    args = parser.parse_args()
    version = args.version

    if args.load_version is None:
        best_model_save_path = './SAC-v{}/logs/best_model.zip'.format(version)
    else:
        best_model_save_path = './SAC-v{}/logs/best_model.zip'.format(args.load_version)

    env = build_env(TASK_NAME, ClASS_NAME, version, RUN_MODE, SIMULATOR, visual=True, ctrl_delay=True)
    model = SAC.load(best_model_save_path, device=torch.device('cuda:0'))

    obs = env.reset()
    total_reward = 0
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        # action = np.array([-10, 30, -75,
        #            10, 30, -75,
        #            -10, 50, -75,
        #            10, 50, -75]) * np.pi / 180

        obs, reward, done, info = env.step(action)

        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()