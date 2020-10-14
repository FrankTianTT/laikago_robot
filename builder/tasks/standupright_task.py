from laikago_task import LaikagoTask

class LaikagoStandUpright(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUpright, self).__init__(mode)
        pass

    def reward(self):
        return self.reward_rotation(self.reward_still() + self.reward_up())