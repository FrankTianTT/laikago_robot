from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math

"""
version 6-1 is the best from then on.
"""

class LaikagoStandFromLieBulletBase(LaikagoTaskBullet):

    def __init__(self, mode='train'):
        super(LaikagoStandFromLieBulletBase, self).__init__(mode,
                                                       init_pose=InitPose.LIE)
        # self.mode = 'no-die'
        self.steps = 0

    def reset(self, env):
        self._env = env
        self.steps = 0

    def update(self):
        self.steps += 1

    @property
    def is_healthy(self):
        return not (self.done_rp_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.3) or
                    self.done_region_bullet(threshold=3))

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 1000:
            return True
        else:
            return False

class LaikagoStandFromLieBullet0(LaikagoStandFromLieBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandFromLieBullet0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_toe_distance(), 1)
        self.add_reward(self.reward_energy(), 3)
        if self.is_healthy:
            return self.get_sum_reward()
        else:
            return self.get_sum_reward() - 1
