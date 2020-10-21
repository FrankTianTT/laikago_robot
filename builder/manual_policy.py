import torch
import argparse
import importlib
import sys
from os.path import abspath, join, dirname

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from builder.gym_env import LaikagoEnv
import builder.tasks_sim as tasks_sim
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", required=True, help="Version of task")

    args = parser.parse_args()
    version = args.version

    standup_task_sim = importlib.import_module('builder.tasks_sim.standup_task_sim')
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
        if 0 < i < 200:
            for j in [1, 4, 7, 10]:
                action[j] += 0.1 * np.pi / 180
        elif 200 < i < 400:
            for j in [1, 4, 7, 10]:
                action[j] -= 0.1 * np.pi / 180
        elif 400 < i < 600:
            for j in [0, 6]:
                action[j] -= 0.1 * np.pi / 180
            for j in [3, 9]:
                action[j] -= 0.1 * np.pi / 180
        elif 600 < i < 800:
            for j in [0, 6]:
                action[j] += 0.1 * np.pi / 180
            for j in [3, 9]:
                action[j] += 0.1 * np.pi / 180

        obs, reward, done, info = env.step(action)

        total_reward += reward
        i += 1
        if done:
            obs = env.reset()
            print('Test reward is {:.3f}.'.format(total_reward))
            total_reward = 0
            i = 0
    env.close()
