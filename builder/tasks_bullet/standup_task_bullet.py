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
        if self.steps > 300:
            return True
        else:
            return self.done_rp_bullet(threshold=30) or self.done_height_bullet(threshold=0.3) or self.done_region_bullet(threshold=3)

class LaikagoStandUpBullet0(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet1, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()


class LaikagoStandUpBullet1_1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet1_1, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_height_bullet(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet2(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet2, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_base_vel_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet3(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet3, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_rpy_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet4(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet4, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_region_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet5(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet5, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_toe_contact(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet5_1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet5_1, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_toe_contact(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet6(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet6, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_distance(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet6_1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet6_1, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_distance(), 1)
        self.add_reward(self.reward_toe_contact(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet6_2(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet6_2, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_distance(), 1)
        self.add_reward(self.reward_toe_contact(), 1)
        self.add_reward(self.reward_rpy_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet7(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet7, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet7_1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet7_1, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_toe_distance(threshold=0.15), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet7_2(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet7_2, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_toe_distance(threshold=0.15), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_rpy_bullet(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet7_3(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet7_3, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_height_bullet(), 1)
        self.add_reward(self.reward_toe_distance(threshold=0.15), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_toe_contact(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet8(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet8, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_toe_contact_soft(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpBullet8_1(LaikagoStandUpBulletBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBullet8_1, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_toe_contact_soft(), 1)
        return self.get_sum_reward()