from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoTurn(LaikagoTask):

    def __init__(self, mode='train', dir=env_constant.TURN_LEFT):
        super(LaikagoTurn, self).__init__(mode)
        self.dir = dir
        pass

    def reward(self):
        return self.reward_turn(self.dir)

    def reset(self, env):
        super(LaikagoTurn, self).reset(env)
        self.fall_timer = 0

    def done(self):
        if self.steps > 1000:
            return True
        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0
