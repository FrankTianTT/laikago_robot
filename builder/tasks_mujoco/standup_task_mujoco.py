from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpMujocoBase(LaikagoTaskMujoco):

    def __init__(self, run_mode='train',
                 reward_mode="with_shaping"):
        super(LaikagoStandUpMujocoBase, self).__init__(run_mode,
                                                       init_pose=InitPose.STAND,
                                                       reward_mode=reward_mode)
        # self.mode = 'no-die'
        self.max_episode_steps = 1000
        self.steps = 0

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=30) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=30) or
                    self.done_height_mujoco(threshold=0.2))

class LaikagoStandUpMujoco0(LaikagoStandUpMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpMujoco0, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=30) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=30) or
                    self.done_height_mujoco(threshold=0.3) or
                    self.done_region_mujoco(threshold=3) or
                    self.done_toe_distance(threshold=0.1) or
                    self.reward_toe_height_mujoco(threshold=0.05))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=30) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=30) + self.reward_height_mujoco(threshold=0.3) + \
              self.reward_region_mujoco(threshold=3) + self.reward_toe_distance(threshold=0.1) + \
              self.reward_toe_height_mujoco(threshold=0.05)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)

class LaikagoStandUpMujoco1(LaikagoStandUpMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpMujoco1, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=30) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=30) or
                    self.done_height_mujoco(threshold=0.2) or
                    self.done_toe_distance(threshold=0.1) or
                    self.reward_toe_contact_long(threshold=14))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=30) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=30) + self.reward_height_mujoco(threshold=0.2) + \
              self.reward_toe_distance(threshold=0.1) + self.reward_toe_contact_long(threshold=14)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)
