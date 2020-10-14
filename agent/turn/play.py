from builder.tasks.turn_task import LaikagoTurn
from builder.gym_env import LaikagoEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch
import numpy as np

task = LaikagoTurn()
env = LaikagoEnv(task=task, visual=True)

model = SAC.load("logs/best_model")

if __name__ == '__main__':
    total_reward = 0
    obs = env.reset()
    for i in range(100000):
        # action, _states = model.predict(obs, deterministic=True)
        action = np.array([0, 40, -75,
                   0, 40, -75,
                   0, 40, -75,
                   0, 40, -75]) * np.pi / 180
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
          obs = env.reset()
          print('Test reward is {:.3f}.'.format(total_reward))
          total_reward = 0
    env.close()