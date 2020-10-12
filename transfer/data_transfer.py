from robot.laikago import Laikago
from transfer import transfer_constant
import collections
import numpy as np


class Transfer(object):

    def __init__(self,
                 pybullet_client,
                 kp=transfer_constant.KP,
                 kd=transfer_constant.KD,
                 torque_limits=transfer_constant.TORQUE_LIMITS,
                 robot_class=Laikago,
                 history_len=transfer_constant.HISTORY_LEN):
        self._pybullet_client = pybullet_client
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        self.laikago = robot_class(pybullet_client)
        self.history_len = history_len
        self.history_observation = collections.deque(maxlen=history_len)

    def step(self, pos_action):
        """

        :param action: 这个action是上层传过来的，应该是position
        :return:
        """
        torque_action = self.position2torque(pos_action)
        obs = self.laikago.step(torque_action)
        self.history_observation.appendleft(obs)
        return self.history_observation

    def reset(self):
        self.laikago.reset()

    def get_observation(self):
        return self.laikago.get_observation()

    def _init_history_observation(self):
        for i in range(self.history_len):
            self.history_observation.appendleft(np.zeros(33))

    def position2torque(self, target_pos, target_vel, pos, vel):
        """
        通过PD控制将位置信号转化为电机的扭矩信号
        :param target_pos: 目标位置
        :param target_vel: 目标速度
        :param pos: 观测位置
        :param vel: 观测速度
        :return: 对应的电机扭矩
        """
        target_pos = np.array(target_pos)
        target_vel = np.array(target_vel)
        pos = np.array(pos)
        vel = np.array(vel)
        additional_torques = 0
        motor_torques = -1 * (self._kp * (pos - target_pos)) - self._kd * (vel - target_vel) + additional_torques
        motor_torques = np.clip(motor_torques, -1*self._torque_limits, self._torque_limits)

        return motor_torques

    def _cal_toe_position(self):
        """Get the robot's foot position in the base frame."""
        foot_positions = []
        toe_link_ids = self.laikago.get_toe_link_ids()
        for toe_link_id in toe_link_ids:
            foot_positions.append(self._link_pos_in_base_frame(toe_link_id))

        return np.array(foot_positions)

    def _link_pos_in_base_frame(self, link_id):
        """
        Computes the link's local position in the robot frame.
        Args:
            robot: A robot instance.
            link_id: The link to calculate its relative position.

        Returns:
            The relative position of the link.
        """
        base_position, base_orientation = self._pybullet_client.getBasePositionAndOrientation(self.laikago.quadruped)
        inverse_translation, inverse_rotation = self._pybullet_client.invertTransform(
            base_position, base_orientation)

        link_state = self._pybullet_client.getLinkState(self.laikago.quadruped, link_id)
        link_position = link_state[0]
        link_local_position, _ = self._pybullet_client.multiplyTransforms(
            inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

        return np.array(link_local_position)
