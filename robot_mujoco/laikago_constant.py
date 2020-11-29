import re
import numpy as np
import os
ROBOT_DIR_NAME = os.path.dirname(__file__)
URDF_DIR_NAME = os.path.join(ROBOT_DIR_NAME, 'laikago_urdf')
URDF_FILE = os.path.join(URDF_DIR_NAME, 'laikago.xml')
# URDF_DIR_NAME = os.path.join(ROBOT_DIR_NAME, 'laikago_model')
# URDF_FILE = os.path.join(URDF_DIR_NAME, 'laikago_toes_limits.urdf')

CAMERA_SETTING = {'camera_distance': 1.0,
                  'camera_yaw': 0,
                  'camera_pitch': -30,
                  'render_width': 480,
                  'render_height': 360}
TIME_STEP = 0.001
ACTION_REPEAT = 20
NUM_MOTORS = 12
DOFS_PER_LEG = 3

ON_RACK_INIT_POSITION = [0, 0, 1]
STAND_INIT_POSITION = [0, 0, 0.43]
LIE_INIT_POSITION = [0, 0, 0.15]

INIT_ORIENTATION = [0, 0, 0, 0]

HIP_NAME_PATTERN = re.compile(r"\w{2}_hip_motor_2_chassis_joint")
UPPER_NAME_PATTERN = re.compile(r"\w{2}_upper_leg_2_hip_motor_joint")
LOWER_NAME_PATTERN = re.compile(r"\w{2}_lower_leg_2_upper_leg_joint")
TOE_NAME_PATTERN = re.compile(r"jtoe\w{2}")

JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

TOE_GEOM_NAME = ["FR_toe", "FL_toe", "RR_toe", "RL_toe"]

TOE_GEOM_ID = [8, 14, 20, 26]

LIE_MOTOR_ANGLES = np.array([0, 88, -155,
                   0, 88, -155,
                   0, 88, -155,
                   0, 88, -155]) * np.pi / 180

STAND_MOTOR_ANGLES = np.array([-10, 30, -75,
                   10, 30, -75,
                   -10, 50, -75,
                   10, 50, -75]) * np.pi / 180

"""
Randomization:随机化分为两部分，一部分是观测的随机，另一部分是动力学参数的随机。
"""

SENSOR_NOISE_STDDEV = {'IMU_angle': 0.001, 'IMU_rate': 0.01, 'motor_angle': 0.001, 'motor_velocity': 0.01}
# relative bound
MASS_BOUND = [-0.2, 0.2]
INERTIA_BOUND = [-0.5, 0.5]
# abstract bound
JOINT_F_BOUND = [0, 0.1]
TOE_F_BOUND = [0.5, 1.25]
G_BOUND = [9.7, 9.9]



OBSERVATION_HISTORY_LEN = 100

from enum import Enum


class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 8.0
HIP_P_GAIN = 220.0
HIP_D_GAIN = 8.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 8.0
KP = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
KD = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]
TORQUE_LIMITS = np.ones(12) * 40

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2