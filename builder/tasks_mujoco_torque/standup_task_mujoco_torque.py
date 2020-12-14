from builder.laikago_task_mujoco import LaikagoTaskMujoco
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpMujocoTorqueBase(LaikagoTaskMujoco):

    def __init__(self,
                 reward_mode='with_shaping',
                 run_mode='train'):
        super(LaikagoStandUpMujocoTorqueBase, self).__init__(run_mode=run_mode,
                                                       reward_mode=reward_mode,
                                                       init_pose=InitPose.STAND)
    @property
    def is_healthy(self):
        return not (self.done_r_mujoco(threshold=30) or
                    self.done_p_mujoco(threshold=30) or
                    self.done_y_mujoco(threshold=30) or
                    self.done_height_mujoco(threshold=0.2))

class LaikagoStandUpMujocoTorque0(LaikagoStandUpMujocoTorqueBase):

    def __init__(self, run_mode='train', reward_mode='without_shaping'):
        super(LaikagoStandUpMujocoTorque0, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)
    def update_reward(self):
        self.add_reward(self.reward_quad_ctrl(), 1)
        self.add_reward(self.reward_height_mujoco(threshold=0.3), 1)

class LaikagoStandUpMujocoTorque1(LaikagoStandUpMujocoTorqueBase):

    def __init__(self, run_mode='train'):
        super(LaikagoStandUpMujocoTorque1, self).__init__(run_mode)
        self.die_if_unhealthy = False

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=30) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=30) + self.reward_height_mujoco(threshold=0.2)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)

class LaikagoStandUpMujocoTorque2(LaikagoStandUpMujocoTorqueBase):

    def __init__(self, run_mode='train'):
        super(LaikagoStandUpMujocoTorque2, self).__init__(run_mode)
        self.die_if_unhealthy = False

    def cal_phi_function(self):
        sum = self.reward_r_mujoco(threshold=30) + self.reward_p_mujoco(threshold=30) + \
              self.reward_y_mujoco(threshold=30) + self.reward_height_mujoco(threshold=0.2)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)