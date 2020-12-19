from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math
import random

class LaikagoStandUpBulletBase(LaikagoTaskBullet):

    def __init__(self,
                 reward_mode='with_shaping',
                 run_mode='train',
                 contact_buffer_length=5):
        super(LaikagoStandUpBulletBase, self).__init__(run_mode=run_mode,
                                                       reward_mode=reward_mode,
                                                       init_pose=InitPose.STAND,
                                                       contact_buffer_length=contact_buffer_length)

class LaikagoStandUpBulletPush(LaikagoStandUpBulletBase):
    def __init__(self,
                 run_mode='train',
                 reward_mode='with_shaping',
                 force=True,
                 max_force=300,
                 force_delay_steps=10):
        super(LaikagoStandUpBulletPush, self).__init__(run_mode=run_mode, reward_mode=reward_mode)
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
                ori = [o[0] + o[1] for o in zip(i, j)]
                self.force_ori.append(ori)

    def _give_force(self):
        if self.now_force is None or self.steps % self.force_delay_steps == 0:
            force_id = random.randint(0, len(self.force_ori) - 1)
            ori = self.force_ori[force_id]
            self.now_force = [f * random.random() * self.max_force for f in ori]
        return self.now_force

    def update(self):
        super(LaikagoStandUpBulletPush, self).update()
        if not self.force:
            return
        else:
            force = self._give_force()
            self._env.transfer.laikago.apply_force(force)

class LaikagoStandUpBullet0(LaikagoStandUpBulletBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpBullet0, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=30) or
                    self.done_p_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1) or
                    self.done_toe_contact_long(threshold=16))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=30) + self.reward_p_bullet(threshold=30) + \
              self.reward_height_bullet(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1) + \
              self.reward_toe_contact_long(threshold=16)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)

class LaikagoStandUpBullet1(LaikagoStandUpBulletBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpBullet1, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode,
                                                    contact_buffer_length=2)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=30) or
                    self.done_p_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1) or
                    self.done_toe_contact_long(threshold=7))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=30) + self.reward_p_bullet(threshold=30) + \
              self.reward_height_bullet(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1) + \
              self.reward_toe_contact_long(threshold=7)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)

class LaikagoStandUpBullet2(LaikagoStandUpBulletBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping'):
        super(LaikagoStandUpBullet2, self).__init__(run_mode=run_mode,
                                                    reward_mode=reward_mode,
                                                    contact_buffer_length=3)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=30) or
                    self.done_p_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.25) or
                    self.done_toe_distance(threshold=0.1) or
                    self.done_toe_contact_long(threshold=9))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=30) + self.reward_p_bullet(threshold=30) + \
              self.reward_height_bullet(threshold=0.25) + \
              self.reward_toe_distance(threshold=0.1) + \
              self.reward_toe_contact_long(threshold=9)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(self.reward_energy(), 1)