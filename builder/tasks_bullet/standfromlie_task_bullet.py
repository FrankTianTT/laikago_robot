from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math

class LaikagoStandFromLieBulletBase(LaikagoTaskBullet):

    def __init__(self,
                 reward_mode='without_shaping',
                 run_mode='train'):
        super(LaikagoStandFromLieBulletBase, self).__init__(run_mode=run_mode,
                                                            reward_mode=reward_mode,
                                                            init_pose=InitPose.LIE)

class LaikagoStandFromLieBullet0(LaikagoStandFromLieBulletBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping',):
        super(LaikagoStandFromLieBullet0, self).__init__(run_mode=run_mode,
                                                         reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=30) or
                    self.done_p_bullet(threshold=30) or
                    self.done_y_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.25) or
                    self.done_region_bullet(threshold=3) or
                    self.done_toe_contact_long(threshold=30) or
                    self.done_toe_distance(threshold=0.2))

    def cal_phi_function(self):
        sum = self.reward_r_bullet(threshold=30) + self.reward_p_bullet(threshold=30) + \
              self.reward_y_bullet(threshold=30) + self.reward_height_bullet(threshold=0.25) + \
              self.reward_toe_contact_soft() * 5 + self.reward_region_bullet(threshold=3) + \
              self.reward_toe_distance(threshold=0.2)
        return sum

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)
