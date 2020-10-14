import re
import numpy as np
import os
ROBOT_DIR_NAME = os.path.dirname(__file__)
URDF_DIR_NAME = os.path.join(ROBOT_DIR_NAME, 'laikago_urdf')
URDF_FILE = os.path.join(URDF_DIR_NAME, 'laikago.urdf')

CAMERA_SETTING = {'camera_distance': 2.0,
                  'camera_yaw': 0,
                  'camera_pitch': -30}
TIME_STEP = 0.002
ACTION_REPEAT = 10
NUM_MOTORS = 12
DOFS_PER_LEG = 3

ON_RACK_INIT_POSITION = [0, 0, 1]
STAND_INIT_POSITION = [0, 0, 0.48]
LIE_INIT_POSITION = [0, 0, 0.15]

INIT_ORIENTATION = [0, 0, 0, 1]

HIP_NAME_PATTERN = re.compile(r"\w{2}_hip_motor_2_chassis_joint")
UPPER_NAME_PATTERN = re.compile(r"\w{2}_upper_leg_2_hip_motor_joint")
LOWER_NAME_PATTERN = re.compile(r"\w{2}_lower_leg_2_upper_leg_joint")
TOE_NAME_PATTERN = re.compile(r"jtoe\w{2}")

MOTOR_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]

LIE_MOTOR_ANGLES = np.array([0, 90, -155,
                   0, 90, -155,
                   0, 180, -155,
                   0, 180, -155]) * np.pi / 180

STAND_MOTOR_ANGLES = np.array([0, 40, -75,
                   0, 40, -75,
                   0, 40, -75,
                   0, 40, -75]) * np.pi / 180

"""
Randomization:随机化分为两部分，一部分是观测的随机，另一部分是动力学参数的随机。
"""

SENSOR_NOISE_STDDEV = {'IMU_angle': 0, 'IMU_rate': 0, 'motor_angle': 0, 'motor_velocity': 0}
# related bound
MASS_BOUND = [0.8, 1.2]
INERTIA_BOUND = [0.5, 1.5]
# abstract bound
JOINT_F_BOUND = [0, 0.1]
TOE_F_BOUND = [0.5, 1.25]

OBSERVATION_HISTORY_LEN = 100

from enum import Enum


class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

ABDUCTION_P_GAIN = 220.0
ABDUCTION_D_GAIN = 0.3
HIP_P_GAIN = 220.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 220.0
KNEE_D_GAIN = 2.0
KP = [ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
            ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN]
KD = [ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
            ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN]
TORQUE_LIMITS = np.ones(12) * 40