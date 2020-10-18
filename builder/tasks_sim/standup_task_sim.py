from laikago_task_sim import LaikagoTaskSim
from builder.laikago_task import InitPose

class LaikagoStandUpSim1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_height_sim() * 3

class LaikagoStandUpSim2(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_height_sim() * 1.5 + self.reward_up() * 0.5

class LaikagoStandUpSim3(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim3, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_height_sim() * 1.5 + self.reward_up() * 0.5 + self.reward_energy() * 0.1

class LaikagoStandUpSim4(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_height_sim() + self.reward_base_vel_sim()


