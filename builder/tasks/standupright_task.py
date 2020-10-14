from laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandUpright(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUpright, self).__init__(mode,
                                                  init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_rotation(self.reward_still() + self.reward_up())

    def reset(self, env):
        super(LaikagoStandUpright, self).reset(env)
        self.fall_timer = 0

    def done(self):
        if self.steps > 1000:
            return True
        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0

        if self.fall_timer > 20:
            return True
