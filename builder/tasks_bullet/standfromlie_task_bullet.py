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

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 300:
            return True
        else:
            return self.done_rp_bullet(threshold=30) or self.done_height_adaptation_bullet()

class LaikagoStandFromLieBullet0(LaikagoStandFromLieBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandFromLieBullet0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()