from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpMujocoBase(LaikagoTaskMujoco):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujocoBase, self).__init__(mode,
                                                       init_pose=InitPose.STAND)
        # self.mode = 'no-die'
        self.max_episode_steps = 1000
        self.steps = 0

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=10) or
                    self.done_y_mujoco(threshold=10) or
                    self.done_height_mujoco(threshold=0.3) or
                    # self.done_region_mujoco(threshold=0.1) or
                    self.done_toe_contact())

class LaikagoStandUpMujoco0(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco0, self).__init__(mode)
        self.die_if_unhealthy = False

    def cal_reward(self):
        self.add_reward(self.reward_energy(), 1)

class LaikagoStandUpMujoco0_1(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco0_1, self).__init__(mode)
        self.die_if_unhealthy = False

    def cal_reward(self):
        self.add_reward(1, 1)

class LaikagoStandUpMujoco1(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco1, self).__init__(mode)
        self.die_if_unhealthy = False

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=60) or
                    self.done_p_mujoco(threshold=60) or
                    self.done_y_mujoco(threshold=60) or
                    self.done_height_mujoco(threshold=0.15) or
                    # self.done_region_mujoco(threshold=0.1) or
                    self.done_toe_contact())

    def cal_reward(self):
        self.add_reward(self.reward_energy(), 1)

class LaikagoStandUpMujoco1_1(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco1_1, self).__init__(mode)
        self.die_if_unhealthy = False

    def cal_reward(self):
        self.add_reward(1, 1)

class LaikagoStandUpMujoco4(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco4, self).__init__(mode)
        self.die_if_unhealthy = True

    @property
    def is_healthy(self):
        return not(self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=10) or
                    self.done_height_mujoco(threshold=0.15) or
                    self.done_region_mujoco(threshold=3))


    def cal_reward(self):
        self.add_reward(self.reward_height_mujoco(threshold=0.3), 1)
        self.add_reward(self.reward_region_mujoco(threshold=0.5), 1)
        self.add_reward(self.reward_energy(), 1)