from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpMujocoBase(LaikagoTaskMujoco):

    def __init__(self, run_mode='train',
                 reward_mode="with_shaping",
                 contact_buffer_length=5):
        super(LaikagoStandUpMujocoBase, self).__init__(run_mode,
                                                       init_pose=InitPose.STAND,
                                                       reward_mode=reward_mode,
                                                       contact_buffer_length=contact_buffer_length)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=30) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=30) or
                    self.done_height_mujoco(threshold=0.2))


class LaikagoStandUpMujoco0(LaikagoStandUpMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpMujoco0, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode,
                                                    contact_buffer_length=3)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=10) or
                    self.done_y_mujoco(threshold=10) or
                    self.done_height_mujoco(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=10) + self.reward_p_mujoco(threshold=10) + \
              self.reward_y_mujoco(threshold=10) + self.reward_height_mujoco(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1)
        return sum / 5

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)

class LaikagoStandUpMujoco0_1(LaikagoStandUpMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpMujoco0_1, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode,
                                                    contact_buffer_length=3)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=10) or
                    self.done_y_mujoco(threshold=10) or
                    self.done_speed_mujoco(threshold=0.1) or
                    self.done_height_mujoco(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=10) + self.reward_p_mujoco(threshold=10) + \
              self.reward_y_mujoco(threshold=10) + self.reward_speed_mujoco(threshold=0.1) + \
              self.reward_height_mujoco(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1)
        return sum / 5

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)