from robot_simulation.laikago import Laikago
from transfer import transfer_constant
from transfer.transfer_constant import InitPose
from transfer.transfer_constant import FL, FR, RL, RR, L1, L2, L3, ROBOT_LENGTH, ROBOT_WIDTH
import collections
import numpy as np


class Transfer(object):

    def __init__(self,
                 init_pose=InitPose.STAND,
                 robot_class=Laikago,
                 visual=False,
                 history_len=transfer_constant.HISTORY_LEN):

        self.laikago = None
        self.robot_class = robot_class
        self.visual = visual
        self.history_len = history_len
        self.history_observation = collections.deque(maxlen=history_len)
        self._init_history_observation()
        self.laikago = self.robot_class(visual=self.visual, init_pose=init_pose)
        self.observation = None

    def step(self, pos_action):
        """

        :param action: 这个action是上层传过来的，应该是position
        :return:
        """
        obs, energy = self.laikago.step(pos_action)
        self.observation = obs
        self.collocation_observation(obs)
        return self.get_env_observation(), energy

    def get_env_observation(self):
        obs = []
        obs.extend(np.array(self.history_observation[0][0:12])/(np.pi))
        obs.extend(np.array(self.history_observation[0][12:24]) / (10 * np.pi))
        for i in range(self.history_len):
            obs.extend(np.array(self.history_observation[i][24:27])/(np.pi))
            obs.extend(np.array(self.history_observation[i][27:30])/(10 * np.pi))
            obs.extend(np.array(self.history_observation[i][30:34]))
            obs.extend(np.array(self.history_observation[i][34:46]) * 2)
        return obs

    def collocation_observation(self, obs):
        """
        收集历史数据，历史数据分别为
        [0:12] motor angle
        [12:24] motor velocity
        [24:27] rpy
        [27:30] dRPY
        [30:34] toe collision
        [34:46] toe position (3*4)
        [46:49] chassis velocity
        :param obs:
        :return:
        """
        toe_position = self.get_toe_position()
        chassis_vel = self.get_chassis_vel_by_toe()
        obs.extend(toe_position)
        obs.extend(chassis_vel)
        self.history_observation.appendleft(obs)

    def reset(self):
        self.laikago.reset(init_reset=False)
        self._init_history_observation()
        return self.get_env_observation()

    def get_observation(self):
        return self.observation

    def _init_history_observation(self):
        for i in range(self.history_len):
            self.history_observation.appendleft(np.zeros(49))

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
            pos.extend(self.compute_toe_position(toe_id=id2id[toe_id], motor_angle=motor_angle).tolist())
        return list(pos)

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
        return ret[: -1].reshape(-1)

    def get_chassis_vel_by_toe(self):
        now_toe_position = self.get_toe_position()
        last_toe_position = self.get_history_toe_position()[0]
        now_toe_height = [now_toe_position[index] for index in [0, 3, 6, 9]]
        lowest_toe_id = np.argmin(now_toe_height)
        return (last_toe_position[lowest_toe_id * 3: lowest_toe_id * 3 + 3] -
               now_toe_position[lowest_toe_id * 3: lowest_toe_id * 3 + 3])/0.02

    def get_history_angle(self):
        history_angle = []
        for obs in self.history_observation:
            history_angle.append(obs[0: 12])
        return np.array(history_angle)

    def get_history_chassis_velocity(self):
        chassis_vel = []
        for obs in self.history_observation:
            chassis_vel.append(obs[46: 49])
        return np.array(chassis_vel)

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
            history_toe_position.append(obs[34: 34+12])
        return np.array(history_toe_position)

