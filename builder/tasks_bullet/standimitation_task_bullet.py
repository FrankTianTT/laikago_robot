from builder.laikago_task_bullet import LaikagoTaskBullet
from builder.laikago_task import InitPose
import math
import numpy as np

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0

class LaikagoStandImitationBulletBase(LaikagoTaskBullet):

    def __init__(self, run_mode='train'):
        super(LaikagoStandImitationBulletBase, self).__init__(run_mode,
                                                              init_pose=InitPose.STAND)
        # self.mode = 'no-die'
        self.steps = 0
        self.imit_action = np.array([-10, 30, -75,
                                     10, 30, -75,
                                     -10, 50, -75,
                                     10, 50, -75]) * np.pi / 180
        self._kp = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
        self._kd = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]
        self._torque_limits = np.ones(12) * 40

    def reset(self, env):
        self._env = env
        self.steps = 0

    def update(self):
        self.steps += 1

    def done(self):
        if self.run_mode== 'no-die':
            return False
        if self.steps > 300:
            return True
        else:
            return self.done_rp_bullet(threshold=30) + self.done_height_bullet(threshold=0.3)

    def reward_imitation(self):
        pos = np.array(self._env.get_history_angle()[0])
        vel = np.array(self._env.get_history_velocity()[0])
        target_pos = self.imit_action
        target_vel = np.zeros(12)
        motor_torques = -1 * (self._kp * (pos - target_pos)) - self._kd * (vel - target_vel)
        reward = - np.sum(np.abs(motor_torques))
        return 2/(1+math.exp(-reward/100))

class LaikagoStandImitationBullet0(LaikagoStandImitationBulletBase):

    def __init__(self, run_mode='train'):
        super(LaikagoStandImitationBullet0, self).__init__(run_mode)

    def reward(self):
        self.add_reward(self.reward_imitation(), 1)
        self.add_reward(self.reward_energy(), 1)
        return self.get_sum_reward()