import math
import random
import numpy as np

class LaikagoTask(object):
    def __init__(self, mode='train'):
        self._env = None
        self.mode = mode

        self.body_pos = None
        self.body_ori = None
        self.body_linear_vel = None
        self.body_ang_vel = None
        self.joint_pos = None
        self.joint_vel = None
        self.joint_tor = None

        self.sum_reward = 0
        self.sum_p = 0

        self.max_r = 0
        return

    def set_env(self, env):
        self._env = env

    def reset(self):
        pass

    def update(self):
        pass

    def done(self):
        return False

    def reward(self):
        return 0

    def precision_cost(self, v, t, m):
        w = math.atanh(math.sqrt(0.95)) / m
        return math.tanh(((v - t) * w) ** 2)

    def reward_up(self):
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        return 1 - self.precision_cost(math.sqrt(roll * roll + pitch * pitch), 0.0, 0.4)

    def reward_still(self):
        chassis_vel = self._env.get_history_chassis_velocity()[0]
        return -math.sqrt(chassis_vel[0] ** 2 + chassis_vel[1] ** 2)

    def reward_rotation(self, r):
        yaw = self._env.get_history_rpy()[0][2]
        k = 1 - self.precision_cost(yaw, 0.0, 0.5)
        return min(k * r, r)