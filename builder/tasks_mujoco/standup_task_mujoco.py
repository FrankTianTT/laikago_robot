from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpMujocoBase(LaikagoTaskMujoco):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujocoBase, self).__init__(mode,
                                                       init_pose=InitPose.STAND)
        # self.mode = 'no-die'
        self.steps = 0

    def reset(self, env):
        self._env = env
        self.steps = 0

    def update(self):
        self.steps += 1

    def done(self):
        if self.mode == 'no-die':
            return False
        if self.steps > 300:
            return True
        else:
            return False # self.done_rp(threshold=30) or self.done_min_stand_high(threshold=0.2)

class LaikagoStandUpMujoco0(LaikagoStandUpMujocoBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpMujoco0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_contact_soft(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()