import math
import random
import numpy as np
from enum import Enum
from builder import env_constant
import collections

# Attention! This class use some function that not provided in REALITY.

class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

class LaikagoTaskSim(object):
    def __init__(self,
                 mode='train',
                 init_pose=InitPose.STAND):
        self._env = None
        self.mode = mode
        self.init_pose = init_pose
        self.steps = 0
        self.history_rpy_diff = collections.deque(maxlen=100)

        self.sum_reward = 0
        self.sum_p = 0
        return

    def add_reward(self, reward, p=1):
        self.sum_reward += reward
        self.sum_p += p

    def get_sum_reward(self):
        reward = self.sum_reward / self.sum_p
        self.sum_reward = 0
        self.sum_p = 0
        return reward

    def normalize_reward(self, reward, min_reward, max_reward):
        return (reward - min_reward)/(max_reward - min_reward)


    def reset(self, env):
        self._env = env
        self.steps = 0
        self.fall_timer = 0
        self.history_rpy_diff.clear()
        pass

    def update(self):
        self.steps += 1
        history_rpy = self._env.get_history_rpy()
        rpy_diff = ((np.array(history_rpy[0]) - np.array(history_rpy[-1]))**2).sum()
        self.history_rpy_diff.appendleft(rpy_diff)

        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0
        pass

    def done(self):
        if self.steps > 1000:
            return True

        if self.fall_timer > 50:
            return True

    def reward(self):
        return 0

    def reward_energy(self):
        energy = self._env.get_energy()
        reward = - energy
        return self.normalize_reward(reward, -10, 0)

    def reward_height_sim(self):
        height = self._env.transfer.laikago.get_position_for_reward()[2]
        reward = height
        return self.normalize_reward(reward, 0, 0.4)

    def reward_region_sim(self):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        reward = - math.sqrt(base_pos[0] ** 2 + base_pos[1] ** 2)
        return self.normalize_reward(reward, -3, 0)

    def reward_base_vel_sim(self):
        base_vel = self._env.transfer.laikago.get_velocity_for_reward()
        reward = - math.sqrt(base_vel[0] ** 2 + base_vel[1] ** 2)
        return self.normalize_reward(reward, -3, 0)

    def reward_rpy_sim(self):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        reward = - (r ** 2 + p ** 2)
        return self.normalize_reward(reward, -1, 0)