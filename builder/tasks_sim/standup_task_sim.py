from laikago_task_sim import LaikagoTaskSim
from builder.laikago_task import InitPose
import math

class LaikagoStandUpSim1(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def done(self):
        if self.steps > 1000:
            return True

        if self.fall_timer > 50:
            return True
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        dis = - math.sqrt(base_pos[0] ** 2 + base_pos[1] ** 2)
        if dis > 1:
            return True

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        self.add_reward(self.reward_height_sim(), 3)
        return self.get_sum_reward()

class LaikagoStandUpSim2(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim2, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_up(), 1)
        self.add_reward(self.reward_height_sim(), 5)
        return self.get_sum_reward()

class LaikagoStandUpSim3(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim3, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        self.add_reward(self.reward_up(), 1)
        self.add_reward(self.reward_height_sim(), 5)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim4(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4, self).__init__(mode,
                                              init_pose=InitPose.STAND)
        self.fall_timer = 0
        pass

    def reward(self):
        return self.reward_height_sim() + self.reward_base_vel_sim()


