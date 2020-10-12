from robot.laikago import Laikago
import collections
import numpy as np

class Transfer(object):
    def __init__(self,
                 pybullet_client,
                 robot_class=Laikago,
                 history_len=3):
        self._pybullet_client = pybullet_client
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

    def position2torque(self, pos):
        """
        通过PD控制将位置信号转化为电机的扭矩信号
        :param pos: 目标位置
        :return: 对应的电机扭矩
        """
        return

    def _cal_toe_position(self):
        pass