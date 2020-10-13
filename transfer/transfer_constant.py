import numpy as np


KP = 25
KD = 0.1
TORQUE_LIMITS = np.ones(12) * 20
HISTORY_LEN = 3
ROBOT_WIDTH, ROBOT_LENGTH = 0.175, 0.4387  # 机器人长宽
L1, L2, L3 = 0.037, 0.25, 0.25  # consistent with the manual
FR, FL, RR, RL = (0, 1, 2, 3)
