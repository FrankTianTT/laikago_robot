import math
import numpy as np

L_HIP_UPPER_BOUND = 60*math.pi /180
L_HIP_LOWER_BOUND = -50*math.pi /180
R_HIP_UPPER_BOUND = 50*math.pi /180
R_HIP_LOWER_BOUND = -60*math.pi /180
UPPER_LEG_UPPER_BOUND = 225*math.pi /180
UPPER_LEG_LOWER_BOUND = -30*math.pi /180
LOWER_LEG_UPPER_BOUND = -35*math.pi /180
LOWER_LEG_LOWER_BOUND = -159*math.pi /180

POSITION_UPPER_BOUND = np.array([R_HIP_UPPER_BOUND, UPPER_LEG_UPPER_BOUND, LOWER_LEG_UPPER_BOUND,
                                 L_HIP_UPPER_BOUND, UPPER_LEG_UPPER_BOUND, LOWER_LEG_UPPER_BOUND,
                                 R_HIP_UPPER_BOUND, UPPER_LEG_UPPER_BOUND, LOWER_LEG_UPPER_BOUND,
                                 L_HIP_UPPER_BOUND, UPPER_LEG_UPPER_BOUND, LOWER_LEG_UPPER_BOUND])
POSITION_LOWER_BOUND = np.array([R_HIP_LOWER_BOUND, UPPER_LEG_LOWER_BOUND, LOWER_LEG_LOWER_BOUND,
                                 L_HIP_LOWER_BOUND, UPPER_LEG_LOWER_BOUND, LOWER_LEG_LOWER_BOUND,
                                 R_HIP_LOWER_BOUND, UPPER_LEG_LOWER_BOUND, LOWER_LEG_LOWER_BOUND,
                                 L_HIP_LOWER_BOUND, UPPER_LEG_LOWER_BOUND, LOWER_LEG_LOWER_BOUND])

DIR_LEFT, DIR_RIGHT = 1.0, -1.0
FOOT_FR, FOOT_FL, FOOT_RR, FOOT_RL = 0, 1, 2, 3

