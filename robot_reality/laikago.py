from robot_reality import laikago_constant
import socket
from struct import pack, unpack
from time import sleep
import numpy as np
import collections

class Laikago(object):
    def __init__(self,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 host=laikago_constant.HOST,
                 port=laikago_constant.PORT,
                 action_size=laikago_constant.ACTION_SIZE,
                 obs_size=laikago_constant.OBS_SIZE):
        self.num_motors = num_motors
        self.num_legs = num_motors / dofs_per_leg
        self.host = host
        self.port = port
        self.action_size = action_size
        self.obs_size = obs_size

        self.obs = None
        self._toe_link_ids = [3, 7, 11, 15]
        self.s = socket.socket()
        self.s.bind((self.host, self.port))
        self.s.listen(1)
        self.init_pos = np.array([-15, 15, -35,
                                  15, 15, -35,
                                  -15, 15, -35,
                                  15, 15, -35]) * np.pi / 180
        print("Waiting for connect.")
        self.c, addr = self.s.accept()
        print("Success connect.")

    def reset(self):
        self.c.send(pack('f' * self.action_size, *self.init_pos))
        obs = self.c.recv(1024)
        self.obs = unpack('f' * self.obs_size, obs)  # 这一步其实也会检查obs的大小是否正确
        return self.transfer(self.obs)

    @staticmethod
    def transfer(self, obs):
        """"Convert the real number contact force to a boolean number"""
        for i in range(4):
            obs[30+i] = 1 if obs[30+i] > 1 else -1
        return obs

    def step(self, target_pos):
        try:
            self.c.send(pack('f' * self.action_size, *target_pos))

            obs = self.c.recv(1024)
            self.obs = unpack('f' * self.obs_size, obs)  # 这一步其实也会检查obs的大小是否正确
            self.obs = self.transfer(obs)
        except ConnectionResetError:
            print('ConnectionReset!')
        return obs, None

    def get_observation(self):
        return self.obs

    def get_toe_link_ids(self):
        return self._toe_link_ids
