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

class LaikagoTaskBullet(object):
    def __init__(self,
                 mode='train',
                 init_pose=InitPose.STAND):
        self._env = None
        self.mode = mode
        self.init_pose = init_pose
        self.sum_reward = 0
        self.sum_p = 0
        return

    def add_reward(self, reward, p=1):
        self.sum_reward += reward * p
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
        pass

    def update(self):
        pass

    def done(self):
        return False

    def reward(self):
        return 0

    def reward_energy(self):
        energy = self._env.get_energy()
        reward = - energy
        return self.normalize_reward(reward, -10, 0)

    def reward_height_bullet(self):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        reward = base_pos[2]
        # print(self.normalize_reward(reward, 0, 0.4))
        return self.normalize_reward(reward, 0, 0.4)

    def reward_region_bullet(self):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        reward = - math.sqrt(base_pos[0] ** 2 + base_pos[1] ** 2)
        return self.normalize_reward(reward, -3, 0)

    def reward_base_vel_bullet(self):
        base_vel = self._env.transfer.laikago.get_velocity_for_reward()
        reward = - math.sqrt(base_vel[0] ** 2 + base_vel[1] ** 2)
        return self.normalize_reward(reward, -3, 0)

    def reward_rpy_bullet(self):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        reward = - (r ** 2 + p ** 2)
        return self.normalize_reward(reward, -1, 0)

    def done_rp_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        # print('done rp: ', max(abs(r * 180/np.pi), abs(p * 180/np.pi)))
        return abs(r) > abs(threshold * np.pi / 180) or abs(p) > abs(threshold * np.pi / 180)

    def done_height_bullet(self, threshold=0.35):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        height = base_pos[2]
        # print('done h: ', height)
        return height < threshold

    def done_region_bullet(self, threshold=0.5):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        x = base_pos[0] ** 2 + base_pos[1] ** 2
        return x > threshold ** 2