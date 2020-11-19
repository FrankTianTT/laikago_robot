from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math
import random

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
        if self.steps > 1000:
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
        if self.steps > 1000:
            return True
        else:
            return self.done_rp_bullet(threshold=50) or self.done_height_bullet(threshold=0.2)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_bullet(), 3)
        return self.get_sum_reward()

class LaikagoStandUpBullet2(LaikagoStandUpBulletBase):

    def __init__(self,
                 mode='train',
                 force=True,
                 max_force=300,
                 force_delay_steps=10):
        super(LaikagoStandUpBullet2, self).__init__(mode)
        self.force = force
        self._get_force_ori()
        self.max_force = max_force
        self.force_delay_steps = force_delay_steps

        self.now_force = None
        return

    def _get_force_ori(self):
        self.force_ori = []
        f_ori = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        for i in f_ori:
            for j in f_ori:
                ori = [o[0]+o[1] for o in zip(i, j)]
                self.force_ori.append(ori)

    def _give_force(self):
        if self.steps % self.force_delay_steps == 0:
            force_id = random.randint(0, len(self.force_ori) - 1)
            ori = self.force_ori[force_id]
            self.now_force = [f * random.random() * self.max_force for f in ori]
        return self.now_force

    def update(self):
        if not self.force:
            return
        else:
            force = self._give_force()
            self._env.transfer.laikago.apply_force(force)

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 1000:
            return True
        else:
            return self.done_rp_bullet(threshold=50) or self.done_height_bullet(threshold=0.2)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_bullet(), 3)
        return self.get_sum_reward()
