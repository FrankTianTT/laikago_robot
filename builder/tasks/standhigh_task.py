from laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandHigh(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandHigh, self).__init__(mode,
                                                  init_pose=InitPose.STAND)
        self.fall_timer = 0
        self.no_move_timer = 0
        pass

    def reward(self):
        return self.reward_stand_high()
