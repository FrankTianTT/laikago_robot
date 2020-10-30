from builder.laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandUpBulletBase(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUpBulletBase, self).__init__(mode,
                                                       init_pose=InitPose.STAND)
        # self.mode = 'no-die'
        self.steps = 0

    def reset(self, env):
        self._env = env
        self.steps = 0

    def update(self):
        self.steps += 1

    def done(self):
        if self.mode=='no-die':
            return False
        if self.steps > 300:
            return True
        else:
            return self.done_rp(threshold=30) or self.done_height(threshold=0.3)