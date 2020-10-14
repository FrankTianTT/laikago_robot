from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoTurn(LaikagoTask):

    def __init__(self, mode='train', dir=env_constant.TURN_LEFT):
        super(LaikagoTurn, self).__init__(mode)
        self.dir = dir
        pass

    def reward(self):
        return self.reward_turn(self.dir)