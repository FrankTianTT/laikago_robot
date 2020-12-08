from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math
import random

class LaikagoRunStraightBulletBase(LaikagoTaskBullet):

    def __init__(self, run_mode='train',
                 reward_mode='with_shaping',):
        super(LaikagoRunStraightBulletBase, self).__init__(run_mode,
                                                           init_pose=InitPose.STAND,
                                                           reward_mode='with_shaping')

class LaikagoRunStraightBullet0(LaikagoRunStraightBulletBase):

    def __init__(self, run_mode='train'):
        super(LaikagoRunStraightBullet0, self).__init__(run_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=45) or
                    self.done_p_bullet(threshold=45) or
                    self.done_y_bullet(threshold=15) or
                    self.done_height_bullet(threshold=0.30))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=45) + self.reward_p_bullet(threshold=45) + \
              self.reward_y_bullet(threshold=15) + self.reward_height_bullet(threshold=0.30) # + \
        return sum

    def update_reward(self):
        self.reward_toe_height_bullet()
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)

class LaikagoRunStraightBullet1(LaikagoRunStraightBulletBase):

    def __init__(self, run_mode='train'):
        super(LaikagoRunStraightBullet1, self).__init__(run_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=45) or
                    self.done_p_bullet(threshold=45) or
                    self.done_y_bullet(threshold=15) or
                    self.done_height_bullet(threshold=0.30))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=45) + self.reward_p_bullet(threshold=45) + \
              self.reward_y_bullet(threshold=15) + self.reward_height_bullet(threshold=0.30)
        return sum

    def update_reward(self):
        self.reward_toe_height_bullet()
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)