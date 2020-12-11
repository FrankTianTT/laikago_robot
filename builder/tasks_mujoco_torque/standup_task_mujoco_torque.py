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
class LaikagoStandUpMujocoTorque0(LaikagoStandUpMujocoTorqueBase):

    def __init__(self, run_mode='train', reward_mode='without_shaping'):
        super(LaikagoStandUpMujocoTorque0, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)
    def update_reward(self):
        self.add_reward(self.reward_quad_ctrl(), 1)
        self.add_reward(self.reward_height_mujoco(threshold=0.3), 1)
