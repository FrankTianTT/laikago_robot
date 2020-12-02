from builder.laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandUpright(LaikagoTask):

    def __init__(self, run_mode='train'):
        super(LaikagoStandUpright, self).__init__(run_mode,
                                                  init_pose=InitPose.STAND)
        self.fall_timer = 0
        self.no_move_timer = 0
        pass

    def reward(self):
        return self.reward_rotation(self.reward_still() + self.reward_up())

    def done(self):
        if self.steps > 1000:
            return True

        if self.fall_timer > 20:
            return True