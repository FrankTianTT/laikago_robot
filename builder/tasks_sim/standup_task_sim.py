from builder.laikago_task_sim import LaikagoTaskSim
from builder.laikago_task import InitPose
import math

class LaikagoStandUpSim0(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim0, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_height_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_height_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim1_1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1_1, self).__init__(mode,
                                                 init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()

class LaikagoStandUpSim1_2(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1_2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_height_sim(), 5)

class LaikagoStandUpSim2(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()

class LaikagoStandUpSim3(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim3, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_base_vel_sim(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()

class LaikagoStandUpSim3_1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim3_1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_base_vel_sim(), 1)
        self.add_reward(self.reward_energy(), 1)
        self.add_reward(self.reward_height_sim(), 5)
        return self.get_sum_reward()

class LaikagoStandUpSim4(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_rpy_sim(), 1)
        self.add_reward(self.reward_height_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim4_1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4_1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_rpy_sim(), 1)
        self.add_reward(self.reward_height_sim(), 2)
        return self.get_sum_reward()

class LaikagoStandUpSim4_2(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4_2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_rpy_sim(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()

class LaikagoStandUpSim5(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim5, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_rpy_sim(), 1)
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()