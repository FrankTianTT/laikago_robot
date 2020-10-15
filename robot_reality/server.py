import socket
from struct import pack, unpack
from time import sleep
import numpy as np

OBS_SIZE = 34
ACTION_SIZE = 12
action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

s = socket.socket()
HOST = '127.0.0.5'
PORT = 12345
s.bind((HOST, PORT))
s.listen(1)

torque = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

while True:
    c, addr = s.accept()
    while True:
        try:
            obs = c.recv(1024)
            obs = unpack('f' * OBS_SIZE, obs)  # 这一步其实也会检查obs的大小是否正确

            pos = np.array(obs[:12])
            vel = np.array(obs[12: 24])
            rpy = np.array(obs[24: 27])
            rpy_rate = np.array(obs[27: 30])
            foot_contact = np.array(obs[30: 34])

            # nn计算target_pos
            target_pos = np.array([
                -15, 15, -35,
                15, 15, -35,
                -15, 15, -35,
                15, 15, -35
            ]) * np.pi / 180

            for i in range(12):
                action[i] = target_pos[i]
            c.send(pack('f' * ACTION_SIZE, *action))

            # sleep(0.01)
        except ConnectionResetError:
            break
    c.close()
