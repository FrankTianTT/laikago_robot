import re


URDF_FILE = './laikago_urdf/laikago.urdf'

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
