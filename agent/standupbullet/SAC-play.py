from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))
from builder.gym_env import LaikagoEnv
import builder.tasks_bullet as tasks_sim
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")

    args = parser.parse_args()
    version = args.version

    best_model_save_path = './SAC-v{}/logs/best_model.zip'.format(version)

    standup_task_sim = importlib.import_module('builder.tasks_bullet.standup_task_sim')
    task = eval('standup_task_sim.LaikagoStandUpSim{}()'.format(version))
    env = LaikagoEnv(task=task, visual=True)
    model = SAC.load(best_model_save_path)

    obs = env.reset()
    total_reward = 0
    for i in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        # print(action * 180 / np.pi)
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