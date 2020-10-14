from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoLiftFoot(LaikagoTask):

    def __init__(self, mode='train', foot = env_constant.FOOT_FR):
        super(LaikagoLiftFoot, self).__init__(mode)
        self.foot = foot
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_rotation(self.reward_lift(self.foot) + 0.1 * self.reward_still() + 0.1 * self.reward_up())
