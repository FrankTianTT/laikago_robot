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
                 ):
        self.num_motors = num_motors
        self.num_legs = num_motors / dofs_per_leg
        self.host = host
        self.port = port

        self._toe_link_ids = [3, 7, 11, 15]
        self.s = socket.socket()
        self.s.bind((self.host, self.port))
        self.s.listen(1)

    def step(self):

        pass

    def reset(self):
        pass

    def get_observation(self):
        pass

    def get_toe_link_ids(self):
        return self._toe_link_ids
