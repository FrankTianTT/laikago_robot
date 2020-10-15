from builder.tasks.standhigh_task import LaikagoStandHigh2
from builder.gym_env import LaikagoEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np

task = LaikagoStandHigh2()
env = LaikagoEnv(task=task, visual=True)

model = SAC.load("logs/best_model")

if __name__ == '__main__':
    total_reward = 0
    obs = env.reset()
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()