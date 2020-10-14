from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoLiftFoot(LaikagoTask):

    def __init__(self, mode='train', foot = env_constant.FOOT_FR):
        super(LaikagoLiftFoot, self).__init__(mode)
        self.foot = foot
        pass

    def reward(self):
        return self.reward_rotation(self.reward_lift(self.foot) + 0.1 * self.reward_still() + 0.1 * self.reward_up())

    def reset(self, env):
        super(LaikagoLiftFoot, self).reset(env)
        self.fall_timer = 0

    def done(self):
        if self.steps > 1000:
            return True
        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0
