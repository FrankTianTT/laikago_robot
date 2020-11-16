# 这个文件是对laikago的机械特性和电气特性的仿真。

import robot_mujoco.laikago_constant as laikago_constant
from robot_bullet.laikago_constant import InitPose
import mujoco_py as mj
import math
import numpy as np
import random
import time

class Laikago(object):
    def __init__(self,
                 visual=False,
                 time_step=laikago_constant.TIME_STEP,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 urdf_filename=laikago_constant.URDF_FILE,
                 init_pose=InitPose.ON_ROCK,
                 obs_delay=False,
                 action_repeat=laikago_constant.ACTION_REPEAT,
                 randomized=True,
                 observation_noise_stdev=laikago_constant.SENSOR_NOISE_STDDEV,
                 observation_history_len=laikago_constant.OBSERVATION_HISTORY_LEN,
                 kp=laikago_constant.KP,
                 kd=laikago_constant.KD,
                 torque_limits=laikago_constant.TORQUE_LIMITS,
                 mass_bound=laikago_constant.MASS_BOUND,
                 inertia_bound=laikago_constant.INERTIA_BOUND,
                 joint_f_bound=laikago_constant.JOINT_F_BOUND,
                 toe_f_bound=laikago_constant.TOE_F_BOUND,
                 g_bound=laikago_constant.G_BOUND,
                 max_motor_angle_change_per_step=laikago_constant.MAX_MOTOR_ANGLE_CHANGE_PER_STEP):
        self.visual = visual
        self.model = mj.load_model_from_path(urdf_filename)
        self.sim = mj.MjSim(self.model)
        self.data = self.sim.data
        self.time_step = time_step
        self.num_motors = num_motors
        self.num_legs = num_motors / dofs_per_leg
        self._urdf_filename = urdf_filename
        self._init_pose = init_pose
        self._self_collision_enabled = self_collision_enabled
        self.action_filter_enabled = action_filter_enabled
        self.obs_delay = obs_delay
        self._action_repeat = action_repeat
        self.randomized = randomized
        self._observation_noise_stdev = observation_noise_stdev
        self.observation_history_len = observation_history_len
        self._kp = kp
        self._kd = kd
        self._torque_limits = torque_limits
        self.mass_bound = mass_bound
        self.inertia_bound = inertia_bound
        self.joint_f_bound = joint_f_bound
        self.toe_f_bound = toe_f_bound
        self.g_bound = g_bound
        self.max_motor_angle_change_per_step = max_motor_angle_change_per_step

        self.now_g = sum(g_bound) / 2
        self.energy = 0
        self._step_counter = 0
        self._last_frame_time = 0
        self._last_observation = np.zeros(34).tolist()

        self._action_filter = self._build_action_filter()
        self._last_action = None
        self.reset(init_reset=True)
        self.receive_observation()




model = mj.load_model_from_path('./laikago_urdf/laikago.xml')
sim = mj.MjSim(model)
viewer = mj.MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    viewer.render()
