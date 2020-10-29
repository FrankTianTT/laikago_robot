import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from builder.gym_env import LaikagoEnv
import builder.tasks_bullet as tasks_sim
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")

    args = parser.parse_args()
    version = args.version

    standup_task_sim = importlib.import_module('builder.tasks_bullet.standup_task_sim')
    task = eval('standup_task_sim.LaikagoStandUpSim{}()'.format(version))
    env = LaikagoEnv(task=task, visual=True)

    obs = env.reset()
    total_reward = 0
    i = 0
    action = np.array([-10, 30, -75,
                       10, 30, -75,
                       -10, 50, -75,
                       10, 50, -75]) * np.pi / 180
    while True:
        step = 0.5
        length = 200
        if 0 * length < i < 1 * length:
            for j in [1, 4, 7, 10]:
                action[j] += step * np.pi / 180
        elif 1 * length < i < 3 * length:
            for j in [1, 4, 7, 10]:
                action[j] -= step * np.pi / 180
        elif 3 * length < i < 4 * length:
            for j in [1, 4, 7, 10]:
                action[j] += step * np.pi / 180
        elif 4 * length < i < 5 * length:
            for j in [0, 3, 6, 9]:
                action[j] -= step * np.pi / 180
        elif 5 * length < i < 7 * length:
            for j in [0, 3, 6, 9]:
                action[j] += step * np.pi / 180
        elif 7 * length < i < 8 * length:
            for j in [0, 3, 6, 9]:
                action[j] -= step * np.pi / 180
        elif 8 * length < i < 9 * length:
            for j in [2, 5, 8, 11]:
                action[j] -= step * np.pi / 180
        elif 9 * length < i < 11 * length:
            for j in [2, 5, 8, 11]:
                action[j] += step * np.pi / 180
        elif 11 * length < i < 12 * length:
            for j in [2, 5, 8, 11]:
                action[j] -= step * np.pi / 180

        obs, reward, done, info = env.step(action)
        total_reward += reward
        i += 1
        if done:
            obs = env.reset()
            print('Test reward is {:.3f}.'.format(total_reward))
            total_reward = 0
            i = 0
    env.close()
