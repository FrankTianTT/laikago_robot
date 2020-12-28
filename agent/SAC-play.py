from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
from builder.env_builder import build_env
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")
    parser.add_argument("-lv", "--load_version")
    parser.add_argument('-t', '--task_name', default='standup')
    parser.add_argument('-m', '--run_mode', default='train')
    parser.add_argument('-s', '--simulator', default='mujoco')

    args = parser.parse_args()
    version = args.version
    task_name = args.task_name
    run_mode = args.run_mode
    simulator = args.simulator

    if args.load_version is None:
        best_model_save_path = './{}/{}/SAC-v{}/logs/best_model.zip'.format(simulator, task_name, version)
    else:
        best_model_save_path = './{}/{}/SAC-v{}/logs/best_model.zip'.format(simulator, task_name, version)

    env = build_env(task_name, version, run_mode, simulator, visual=True, ctrl_delay=True)
    model = SAC.load(best_model_save_path, device=torch.device('cuda:0'))

    obs = env.reset()
    total_reward = 0
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        # action = np.array([-10, 30, -75,
        #            10, 30, -75,
        #            -10, 50, -75,
        #            10, 50, -75]) * np.pi / 180
        # action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(info['energy'])

        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()