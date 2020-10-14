import math
import random
import numpy as np
from enum import Enum


class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

class LaikagoTask(object):
    def __init__(self,
                 mode='train',
                 init_pose=InitPose.STAND):
        self._env = None
        self.mode = mode

        self.init_pose = init_pose
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
        return 1 - self.precision_cost(math.sqrt(roll ** 2 + pitch ** 2), 0.0, 0.4)

    def reward_still(self):
        chassis_vel = self._env.get_history_chassis_velocity()[0]
        return -math.sqrt(chassis_vel[0] ** 2 + chassis_vel[1] ** 2)

    def reward_rotation(self, r):
        yaw = self._env.get_history_rpy()[0][2]
        k = 1 - self.precision_cost(yaw, 0.0, 0.5)
        return min(k * r, r)

    def reward_turn(self, dir):
        yaw = self._env.get_history_rpy()[0][2]
        return dir * yaw + 0.1 * self.reward_up()

    def reward_lift(self, foot):
        toe_height = []
        for i in range(0, 4):
            toe_height.append(self._env.get_history_toe_position()[0][3*i+2])
        # print(toe_height)
        h = toe_height[foot] - min(toe_height)
        return min(1, h)
