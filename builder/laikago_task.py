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
