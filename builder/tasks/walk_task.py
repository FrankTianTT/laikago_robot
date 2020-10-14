from laikago_task import LaikagoTask
from builder import env_constant

class LaikagoWalk(LaikagoTask):

    def __init__(self, mode='train', dir = env_constant.WALK_FORWARD):
        super(LaikagoWalk, self).__init__(mode)
        self.dir = dir
        pass

    def reward(self):
        return self.reward_walk(self.dir)