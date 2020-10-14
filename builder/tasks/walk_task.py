from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoWalk(LaikagoTask):

    def __init__(self, mode='train', direction=env_constant.WALK_FORWARD):
        super(LaikagoWalk, self).__init__(mode)
        self.direction = direction
        pass

    def reward(self):
        return self.reward_walk(self.direction)

    def reset(self, env):
        super(LaikagoWalk, self).reset(env)
        self.fall_timer = 0

    def done(self):
        if self.steps > 1000:
            return True
        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0
