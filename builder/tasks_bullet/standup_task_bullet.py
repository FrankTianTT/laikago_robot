from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math

"""
version 6-1 is the best from then on.
"""

class LaikagoStandUpBulletBase(LaikagoTaskBullet):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBulletBase, self).__init__(mode,
                                                       init_pose=InitPose.STAND)
        # self.mode = 'no-die'
        self.steps = 0

    def reset(self, env):
        self._env = env
        self.steps = 0

    def update(self):
        self.steps += 1

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 100:
            return True
        else:
            return self.done_rp_bullet(threshold=30) or self.done_height_bullet(threshold=0.3) or self.done_region_bullet(threshold=3)

class LaikagoStandUpBullet0(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_toe_distance(threshold=0.15), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet1, self).__init__(mode)

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 100:
            return True
        else:
            return self.done_rp_bullet(threshold=50) or self.done_height_bullet(threshold=0.2)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_bullet(), 3)
        return self.get_sum_reward()
