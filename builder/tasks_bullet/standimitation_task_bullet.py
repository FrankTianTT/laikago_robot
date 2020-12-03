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

    def __init__(self,
                 reward_mode='without_shaping',
                 run_mode='train'):
        super(LaikagoStandImitationBulletBase, self).__init__(run_mode=run_mode,
                                                              reward_mode=reward_mode,
                                                              init_pose=InitPose.LIE)
        self.imitation_action = np.array([-10, 30, -75,
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

class LaikagoStandImitationBullet0(LaikagoStandImitationBulletBase):

    def __init__(self, run_mode='train', reward_mode='with_shaping',):
        super(LaikagoStandImitationBullet0, self).__init__(run_mode=run_mode,
                                                           reward_mode=reward_mode)

    @property
    def is_healthy(self):
        return not (self.done_r_bullet(threshold=30) or
                    self.done_p_bullet(threshold=30) or
                    self.done_y_bullet(threshold=30) or
                    self.done_height_bullet(threshold=0.25) or
                    self.done_region_bullet(threshold=3) or
                    self.done_toe_contact_long(threshold=30) or
                    self.done_toe_distance(threshold=0.2))

    def cal_phi_function(self):
        pos = np.array(self._env.get_history_angle()[0])
        vel = np.array(self._env.get_history_velocity()[0])
        target_pos = self.imitation_action
        target_vel = np.zeros(12)
        motor_torques = -1 * (self._kp * (pos - target_pos)) - self._kd * (vel - target_vel)
        return 10 / np.sum(np.abs(motor_torques))

    def update_reward(self):
        if self.is_healthy:
            self.add_reward(1, 1)