import numpy as np


KP = 1
KD = 1
TORQUE_LIMITS = np.zeros(12)
HISTORY_LEN = 3
ROBOT_WIDTH, ROBOT_LENGTH = 0.175, 0.4387  # 机器人长宽
L1, L2, L3 = 0.037, 0.25, 0.25  # consistent with the manual
FR, FL, RR, RL = (0, 1, 2, 3)
