import socket
from struct import pack, unpack
from time import sleep
import numpy as np

OBS_SIZE = 34
ACTION_SIZE = 12
action = [0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
      obs = unpack('f'*OBS_SIZE, obs)
      pos = np.array(obs[:12])
      vel = np.array(obs[12: 24])
      target_pos = np.array([-0.3, 0.35, -0.7, 0.3, 0.35, -0.7, -0.3, 0.35, -0.7, 0.3, 0.35, -0.7])
      torque = np.zeros(12)
      # torque = (target_pos-pos)*15 - (vel-0)*1
      # torque = torque.clip(-15, 15)
      
      # print(unpack('f'*OBS_SIZE, obs))
      # nn
      for i in range(12):
          action[i] = torque[i]
      print(action)
      c.send(pack('f'*ACTION_SIZE, *action))
      sleep(0.01)
    except ConnectionResetError:
      break
  c.close()
