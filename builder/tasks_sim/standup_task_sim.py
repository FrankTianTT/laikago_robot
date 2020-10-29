from builder.laikago_task_sim import LaikagoTaskSim
from builder.laikago_task import InitPose
import math


class LaikagoStandUpSimBase(LaikagoTaskSim):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSimBase, self).__init__(mode,
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
            return self.done_rp_sim() or self.done_height_sim() or self.done_region_sim()

class LaikagoStandUpSim0(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim0, self).__init__(mode)

    def reward(self):
        self.add_reward(1, 1)
        return self.get_sum_reward()

class LaikagoStandUpSim1(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim1, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_height_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim2(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim2, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim3(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim3, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_base_vel_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim4(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim4, self).__init__(mode)
    def reward(self):
        self.add_reward(self.reward_rpy_sim(), 1)
        return self.get_sum_reward()

class LaikagoStandUpSim5(LaikagoStandUpSimBase):

    def __init__(self, mode='train'):
        super(LaikagoStandUpSim5, self).__init__(mode)

    def reward(self):
        self.add_reward(self.reward_region_sim(), 1)
        return self.get_sum_reward()