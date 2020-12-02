import socket
from struct import pack, unpack
import numpy as np
from collections import deque
from load_policy import load_model, predict
from kinematics import get_toe_position
import torch
from torch.nn import Sequential, ReLU, Linear, Flatten, Tanh


OBS_SIZE = 34
ACTION_SIZE = 12
action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

s = socket.socket()
HOST = '127.0.0.1'
PORT = 1235
s.bind((HOST, PORT))
s.listen(1)

torque = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

init_pos = np.array([-10, 30, -75,
                   10, 30, -75,
                   -10, 50, -75,
                   10, 50, -75]) * np.pi / 180

obs_que = deque(maxlen=46*3)
# model = load_model()

model = Sequential(
    Flatten(),
    Linear(138, 256),
    ReLU(),
    Linear(256, 256),
    ReLU(),
    Linear(256, 12),
    Tanh()
)

model.load_state_dict(torch.load('./policy.pt'))


while True:
    c, addr = s.accept()
    c.send(pack('f' * ACTION_SIZE, *init_pos))
    t = 0

    while True:
        try:
            obs = c.recv(1024)
            obs = unpack('f' * OBS_SIZE, obs)  # 这一步其实也会检查obs的大小是否正确

            pos = np.array(obs[:12]) / np.pi
            vel = np.array(obs[12: 24]) / (10 * np.pi)
            rpy = np.array(obs[24: 27]) / np.pi
            rpy_rate = np.array(obs[27: 30]) / (10 * np.pi)
            foot_contact = np.array(obs[30: 34])
            foot_contact[foot_contact > 1] = 1
            foot_contact[foot_contact < 1] = -1
            toe_position = np.array(get_toe_position(obs[:12])) * 2

            obs_que.extend(pos)
            obs_que.extend(vel)
            obs_que.extend(rpy)
            obs_que.extend(rpy_rate)
            obs_que.extend(foot_contact)
            obs_que.extend(toe_position)

            if len(obs_que) < obs_que.maxlen:
                target_pos = np.array([-10, 30, -75,
                       10, 30, -75,
                       -10, 50, -75,
                       10, 50, -75]) * np.pi / 180
                for i in range(ACTION_SIZE):
                    target_pos[i] = obs[i]
            else:
                # nn计算target_pos
                target_pos = np.array([-10, 30, -75,
                       10, 30, -75,
                       -10, 50, -75,
                       10, 50, -75]) * np.pi / 180
                # target_pos = predict(model, obs_que)

            # target_pos = np.array([-10, 30, -75,
            #        10, 30, -75,
            #        -10, 50, -75,
            #        10, 50, -75]) * np.pi / 180

            c.send(pack('f' * ACTION_SIZE, *target_pos))
            print("Send action")
            # sleep(0.01)
        except ConnectionResetError:
            break
    c.close()
