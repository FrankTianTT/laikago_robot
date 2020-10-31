from builder.laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandUpBase(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBase, self).__init__(mode,
                                                 init_pose=InitPose.STAND)
        self.mode = 'no-die'
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
            return self.done_rp(threshold=30) or self.done_min_stand_high(threshold=0.2)

class LaikagoStandUp0(LaikagoStandUpBase):
    def __init__(self, mode='train'):
        super(LaikagoStandUp0, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_toe_contact_soft(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()