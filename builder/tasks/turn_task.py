from builder.laikago_task import LaikagoTask
from builder import env_constant

class LaikagoTurn(LaikagoTask):

    def __init__(self, run_mode='train', dir=env_constant.TURN_LEFT):
        super(LaikagoTurn, self).__init__(run_mode)
        self.dir = dir
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_turn(self.dir)
