from builder.laikago_task import LaikagoTask
from builder import env_constant

class LaikagoWalk(LaikagoTask):

    def __init__(self, run_mode='train', direction=env_constant.WALK_FORWARD):
        super(LaikagoWalk, self).__init__(run_mode)
        self.direction = direction
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_walk(self.direction)
