from robot.laikago import Laikago
from transfer import transfer_constant
from transfer.transfer_constant import FL, FR, RL, RR, L1, L2, L3, ROBOT_LENGTH, ROBOT_WIDTH
import collections
import numpy as np


class Transfer(object):

    def __init__(self,
                 kp=transfer_constant.KP,
                 kd=transfer_constant.KD,
                 torque_limits=transfer_constant.TORQUE_LIMITS,
                 robot_class=Laikago,
                 visual=False,
                 history_len=transfer_constant.HISTORY_LEN):
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        self.laikago = None
        self.robot_class = robot_class
        self.visual = visual
        self.history_len = history_len
        self.history_observation = collections.deque(maxlen=history_len)
        self.laikago = self.robot_class(visual=self.visual)

    def step(self, pos_action):
        """

        :param action: 这个action是上层传过来的，应该是position
        :return:
        """
        torque_action = self.position2torque(pos_action)
        obs = self.laikago.step(torque_action)
        self.collocation_observation(obs)
        print(self.get_toe_position())
        return self.history_observation

    def collocation_observation(self, obs):
        toe_position = self.get_toe_position()
        obs = obs.extend(toe_position)
        self.history_observation.appendleft(obs)

    def reset(self):
        self.laikago.reset(init_reset=False)

    def get_observation(self):
        return self.laikago.get_observation()

    def _init_history_observation(self):
        for i in range(self.history_len):
            self.history_observation.appendleft(np.zeros(34))

    def position2torque(self, target_pos, target_vel=np.zeros(12)):
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
        assert len(self.get_observation()) == 34
        pos = np.array(self.get_observation()[0: 12])
        vel = np.array(self.get_observation()[12: 24])
        # print('pos:', pos)
        # print('vel:', vel)
        additional_torques = 0
        motor_torques = -1 * (self._kp * (pos - target_pos)) - self._kd * (vel - target_vel) + additional_torques
        motor_torques = np.clip(motor_torques, -1*self._torque_limits, self._torque_limits)

        return motor_torques

    @staticmethod
    def get_transform_matrix(alpha, a, d, theta):
        """
        Get transform matrix by DH parameters
        Args:
            alpha: alpha_{i-1}
            a: a_{i-1}
            d: d_i
            theta: theta_i
        Return:
            Transform matrix
        """
        matrix = [
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
            [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha), np.cos(alpha), np.cos(alpha)*d],
            [0, 0, 0, 1]
        ]
        return np.array(matrix)

    @staticmethod
    def rotate_matrix(axis, theta):
        if axis == 'x':
            matrix = [
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ]
        elif axis == 'y':
            matrix = [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ]
        elif axis == 'z':
            matrix = [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        else:
            assert 0
        return np.array(matrix)

    @staticmethod
    def translation_matrix(dx, dy, dz):
        matrix = [
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]
        ]
        return np.array(matrix)
    def get_toe_position(self):
        pos = []
        toe_links = self.laikago.get_toe_link_ids()
        motor_angle = self.get_observation()[0:12]
        id2id = {3: 0, 7: 1, 11: 2, 15: 3}
        for toe_id in toe_links:
            pos.append(self.compute_toe_position(toe_id=id2id[toe_id], motor_angle=motor_angle))
        return pos

    def compute_toe_position(self, toe_id, motor_angle):
        """
        Compute the position of foot
        :param toe_id: the index of legs
        :param motor_angle: current motor angle
        :return: The position in base frame
        """
        motor_angle = motor_angle[toe_id * 3: toe_id * 3 + 3] # current angle
        matrices = []
        flag = 1  # 1 if left, -1 if right

        if toe_id == FR:
            matrices.append(self.translation_matrix(ROBOT_LENGTH/2, -ROBOT_WIDTH/2, 0))  # 从躯干中心坐标系平移到髋
        elif toe_id == FL:
            matrices.append(self.translation_matrix(ROBOT_LENGTH/2, ROBOT_WIDTH/2, 0))
        elif toe_id == RR:
            matrices.append(self.translation_matrix(-ROBOT_LENGTH/2, -ROBOT_WIDTH/2, 0))
        elif toe_id == RL:
            matrices.append(self.translation_matrix(-ROBOT_LENGTH/2, ROBOT_WIDTH/2, 0))
        else:
            assert 0

        matrices.extend([
            self.rotate_matrix('x', motor_angle[0]),           # 旋转使y轴与大腿电机垂直
            self.translation_matrix(0, flag*L1, 0),            # 从髋电机转移到大腿电机
            self.rotate_matrix('y', np.pi/2+motor_angle[1]),   # 旋转到x轴与大腿同向
            self.translation_matrix(L2, 0, 0),                 # 沿大腿平移
            self.rotate_matrix('y', -np.pi/6+motor_angle[2]),  # 在腿关节旋转，直到x轴与小腿同向
            self.translation_matrix(L3, 0, 0)                  # 沿小腿平移
        ])
        matrix = matrices[0]
        for tmp_m in matrices[1:]:
            matrix = np.matmul(matrix, tmp_m)  # 动坐标系，矩阵顺序应该是从左往右
        pos = np.array([[0], [0], [0], [1]])
        ret = np.matmul(matrix, pos)
        assert ret[-1] == 1

        return ret[: -1]

    def get_history_angle(self):
        history_angle = []
        for obs in self.history_observation:
            history_angle.append(obs[0: 12])
        return np.array(history_angle)

    def get_history_chassis_velocity(self):
        return np.zeros((3, 3))

    def get_history_velocity(self):
        history_vel = []
        for obs in self.history_observation:
            history_vel.append(obs[12: 24])
        return np.array(history_vel)

    def get_history_rpy(self):
        history_rpy = []
        for obs in self.history_observation:
            history_rpy.append(obs[24: 27])
        return np.array(history_rpy)

    def get_history_rpy_rate(self):
        history_rpy_rate = []
        for obs in self.history_observation:
            history_rpy_rate.append(obs[27: 30])
        return np.array(history_rpy_rate)

    def get_history_toe_collision(self):
        history_toe_collision = []
        for obs in self.history_observation:
            history_toe_collision.append(obs[30: 34])
        return np.array(history_toe_collision)

    def get_history_toe_position(self):
        history_toe_position = []
        for obs in self.history_observation:
            history_toe_position.append(obs[30: 34])
        return np.array(history_toe_position)
