from laikago_task import LaikagoTask
from builder.laikago_task import InitPose

class LaikagoStandUp1(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUp1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_min_stand_high() * 4

class LaikagoStandUp2(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUp2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_min_stand_high() * 2 + self.reward_rotation(self.reward_still() + self.reward_up()) * 0.5

class LaikagoStandUp3(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUp3, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_min_stand_high() * 2.5 + self.reward_rotation(self.reward_still() + self.reward_up()) * 0.5 \
               + self.reward_energy() * 0.1

class LaikagoStandUp4(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUp4, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_average_stand_high() * 4

class LaikagoStandUp5(LaikagoTask):

    def __init__(self, mode='train'):
        super(LaikagoStandUp5, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_average_stand_high() * 2 + self.reward_rotation(self.reward_still() + self.reward_up()) * 0.5