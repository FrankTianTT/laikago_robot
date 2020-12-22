from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoRunStraightMujocoBase(LaikagoTaskMujoco):

    def __init__(self, run_mode='train',
                 reward_mode='with_shaping',):
        super(LaikagoRunStraightMujocoBase, self).__init__(run_mode,
                                                           init_pose=InitPose.STAND,
                                                           reward_mode='with_shaping')

class LaikagoRunStraightMujoco1(LaikagoRunStraightMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoRunStraightMujoco1, self).__init__(run_mode=run_mode,
                                                        reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=10) or
                    self.done_x_velocity(threshold=0.5) or
                    self.done_y_velocity(threshold=0.1) or
                    self.done_height_mujoco(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=10) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=10) + self.reward_x_velocity(threshold=0.5) + \
              self.reward_y_velocity(threshold=0.1) + self.reward_height_mujoco(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1)
        return sum / 7

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)

class LaikagoRunStraightMujoco2(LaikagoRunStraightMujocoBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoRunStraightMujoco2, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=10) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=10) or
                    self.done_height_mujoco(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1))

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=10) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=10) + self.reward_height_mujoco(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1)
        return sum / 5

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(self.reward_x_velocity(threshold=3), 1)
            self.add_reward(self.reward_y_velocity(threshold=0.1), 1)
