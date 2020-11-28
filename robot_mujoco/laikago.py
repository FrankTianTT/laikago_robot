# 这个文件是对laikago的机械特性和电气特性的仿真。

import robot_mujoco.laikago_constant as laikago_constant
from robot_bullet.laikago_constant import InitPose
import mujoco_py as mj
import math
import numpy as np
import random
import time
from mujoco_py.generated import const as mj_const


class Laikago(object):
    def __init__(self,
                 visual=False,
                 time_step=laikago_constant.TIME_STEP,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 urdf_filename=laikago_constant.URDF_FILE,
                 init_pose=InitPose.ON_ROCK,
                 self_collision_enabled=False,
                 action_filter_enabled=False,
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
        self.time_step = time_step
        self.visual = visual
        self.model = mj.load_model_from_path(urdf_filename)
        self.sim = mj.MjSim(self.model)
        self.model.opt.timestep = self.time_step
        if self.visual:
            self.viewer = mj.MjViewer(self.sim)
            self.viewer.cam.type = mj_const.CAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.model.body_name2id('trunk')
        # print('init trunk mass:', self.sim.model.body_mass[1])
        self.data = self.sim.data
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

        # self._action_filter = self._build_action_filter()
        self._last_action = None
        self.reset(init_reset=True)
        self._base_rotation_rpy = [0., 0., 0.]
        self._last_base_rotation_rpy = [0., 0., 0.]
        self.receive_observation()

    def reset(self, init_reset=False):
        if init_reset:
            self._record_mass_info_from_urdf()
            self._record_inertia_info_from_urdf()
        self.sim.reset()
        init_qpos = []
        if self._init_pose.value == laikago_constant.InitPose.LIE.value:
            init_qpos.extend(laikago_constant.LIE_INIT_POSITION)
            for joint, rad in zip(laikago_constant.JOINT_NAMES, laikago_constant.LIE_MOTOR_ANGLES):
                self.sim.data.set_joint_qpos(joint, rad)

        elif self._init_pose.value == laikago_constant.InitPose.STAND.value:
            init_qpos.extend(laikago_constant.STAND_INIT_POSITION)
            for joint, rad in zip(laikago_constant.JOINT_NAMES, laikago_constant.STAND_MOTOR_ANGLES):
                self.sim.data.set_joint_qpos(joint, rad)
        elif self._init_pose.value == laikago_constant.InitPose.ON_ROCK.value:
            raise NotImplementedError
        else:
            raise RuntimeError('Unknown initial pose')
        init_qpos.extend(laikago_constant.INIT_ORIENTATION)
        self.sim.data.set_joint_qpos('root', init_qpos)
        self.sim.forward()

        if self.visual:
            self.viewer.render()
        if self.randomized:
            self.randomize()
        return

    def step(self, action):
        """Steps simulation."""

        self.energy = 0
        if self.action_filter_enabled:
            action = self._filter_action(action)

        for i in range(self._action_repeat):
            proc_action = self._smooth_action(action, i)
            self._step_internal(proc_action)
            if self.visual:
                self.viewer.render()

            obs = self.get_observation()

        self._step_counter += 1
        self._last_action = action
        return obs, self.energy * self.time_step / self._action_repeat

    def _filter_action(self, action):
        raise NotImplementedError

    def _smooth_action(self, action, substep_count):
        if self._last_action is not None:
            prev_action = self._last_action
        else:
            prev_action = self.get_true_motor_angles()

        lerp = float(substep_count + 1) / self._action_repeat
        proc_action = prev_action + lerp * (action - prev_action)
        return proc_action

    def _clip_action(self, action):
        current_motor_angles = np.array(self.get_true_motor_angles())
        action = np.clip(action,
                         current_motor_angles - self.max_motor_angle_change_per_step,
                         current_motor_angles + self.max_motor_angle_change_per_step)
        return action

    def _step_internal(self, pos_action):
        clipped_pos_action = self._clip_action(pos_action)
        torque_action = self.position2torque(clipped_pos_action)
        self._set_torque_control(torque_action)
        self.sim.step()
        self.receive_observation()
        self.energy += np.sum(np.abs(np.array(torque_action) * self.get_true_motor_angles()))
        return

    def _set_torque_control(self, torque):
        for i in range(len(torque)):
            self.sim.data.ctrl[i] = torque[i]

    def position2torque(self, target_pos, target_vel=np.zeros(12)):
        """
        通过PD控制将位置信号转化为电机的扭矩信号
        :param target_pos: 目标位置
        :param target_vel: 目标速度
        :param pos: 观测位置
        :param vel: 观测速度
        :return: 对应的电机扭矩
        """
        target_pos = np.array(target_pos)
        target_vel = np.array(target_vel)
        pos = np.array(self.get_motor_angles())
        vel = np.array(self.get_motor_velocities())
        additional_torques = 0
        motor_torques = -1 * (self._kp * (pos - target_pos)) - self._kd * (vel - target_vel) + additional_torques
        motor_torques = np.clip(motor_torques, -1 * self._torque_limits, self._torque_limits)

        return motor_torques

    def get_urdf_file(self):
        return self._urdf_filename

    def _get_default_init_position(self):
        if self._init_pose.value == InitPose.ON_ROCK.value:
            return laikago_constant.ON_RACK_INIT_POSITION
        elif self._init_pose.value == InitPose.STAND.value:
            return laikago_constant.STAND_INIT_POSITION
        else:
            return laikago_constant.LIE_INIT_POSITION

    def _get_default_init_orientation(self):
        return laikago_constant.INIT_ORIENTATION

    def get_observation(self):
        observation = []
        observation.extend(self.get_motor_angles())                 # [0, 12]
        observation.extend(self.get_motor_velocities())             # [12, 24]
        observation.extend(self.get_base_roll_pitch_yaw())          # [24, 27]
        observation.extend(self.get_base_roll_pitch_yaw_rate())     # [27, 30]
        observation.extend(self.get_toe_contacts())                 # [30, 34]
        # print('observation', observation)
        return observation

    def receive_observation(self):
        self._joint_pos = [self.sim.data.get_joint_qpos(joint) for joint in laikago_constant.JOINT_NAMES]
        self._joint_vel = [self.sim.data.get_joint_qvel(joint) for joint in laikago_constant.JOINT_NAMES]
        self._base_rotation_mat = self.sim.data.get_body_xmat('trunk')
        self._last_base_rotation_rpy = self._base_rotation_rpy
        self._base_rotation_rpy = self._get_rpy_from_mat(self._base_rotation_mat)

    def get_motor_angles(self):
        return self._add_sensor_noise(np.array(self.get_true_motor_angles()),
            self._observation_noise_stdev['motor_angle'])

    def get_motor_velocities(self):
        return self._add_sensor_noise(
            np.array(self.get_true_motor_velocities()),
            self._observation_noise_stdev['motor_velocity'])

    def get_base_roll_pitch_yaw(self):
        return self._add_sensor_noise(self.get_true_base_roll_pitch_yaw(),
            self._observation_noise_stdev['IMU_angle'])

    def get_base_roll_pitch_yaw_rate(self):
        return self._add_sensor_noise(self.get_true_base_roll_pitch_yaw_rate(),
            self._observation_noise_stdev['IMU_rate'])

    def get_toe_contacts(self):
        contacts = [-1, -1, -1, -1]
        for i in range(self.data.ncon):
            if self.data.contact[i].geom2 in laikago_constant.TOE_GEOM_ID:
                contacts[(self.data.contact[i].geom2 - 8) // 6] = 1
        # print('contacts:', contacts)
        return contacts

    def get_true_motor_angles(self):
        return self._joint_pos

    def get_true_motor_velocities(self):
        return self._joint_vel

    def get_true_base_roll_pitch_yaw(self):
        orientation_mat = self._base_rotation_mat
        roll_pitch_yaw = self._get_rpy_from_mat(orientation_mat)
        return roll_pitch_yaw

    def get_true_base_roll_pitch_yaw_rate(self):
        rpy_rate = (self._base_rotation_rpy - self._last_base_rotation_rpy) / self.time_step
        # print('rpy_rate', rpy_rate)
        return np.array(rpy_rate)

    def _add_sensor_noise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(
            scale=noise_stdev, size=sensor_values.shape)
        return observation

    def _get_rpy_from_mat(self, R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        return np.array([roll, pitch, yaw])

    def _get_rpy_from_quat(self, Q):
        [w, x, y, z] = Q
        sqw = w * w
        sqx = x * x
        sqy = y * y
        sqz = z * z
        yaw = math.atan2(2.0 * (x * y + z * w), (sqx - sqy - sqz + sqw))
        yaw = self._rpy_correction(yaw)
        roll = math.atan2(2.0 * (y * z + x * w), (-sqx - sqy + sqz + sqw))
        roll = self._rpy_correction(roll)
        pitch = math.asin(-2.0 * (x * z - y * w))
        pitch = self._rpy_correction(pitch)
        return np.array([roll, pitch, yaw])

    def _rpy_correction(self, x):
        if x > math.pi:
            return x - 2 * math.pi
        elif x < -np.pi:
            return x + 2 * math.pi
        else:
            return x

    def _record_mass_info_from_urdf(self):
        self._body_mass_urdf = []
        for body_id in range(self.model.nbody):
            self._body_mass_urdf.append(self.model.body_mass[body_id])
        # print('body mass', self._body_mass_urdf)

    def _record_inertia_info_from_urdf(self):
        self._body_inertia_urdf = []
        for body_id in range(self.model.nbody):
            self._body_inertia_urdf.append(self.model.body_inertia[body_id])

    def set_body_mass(self, body_name, value):
        body_id = self.model.body_name2id(body_name)
        self.model.body_mass[body_id] = value

    def set_body_inertia(self, body_name, value):
        body_id = self.model.body_name2id(body_name)
        self.model.body_inertia[body_id] = value

    def set_all_body_mass(self, mass_list):
        assert len(mass_list) == self.model.nbody
        for i in range(1, self.model.nbody):         # body 0 is the worldbody
            self.model.body_mass[i] = mass_list[i]

    def set_all_body_inertia(self, inertia_list):
        assert len(inertia_list) == self.model.nbody
        for i in range(1, self.model.nbody):         # body 0 is the worldbody
            self.model.body_inertia[i] = inertia_list[i]

    def randomize(self):
        self.randomize_body_mass()
        self.randomize_body_inertia()
        self.randomize_gravity()

    def randomize_body_mass(self):
        body_mass = self._body_mass_urdf
        randomized_body_mass = random.uniform(
            np.array(body_mass) * (1.0 + self.mass_bound[0]),
            np.array(body_mass) * (1.0 + self.mass_bound[1]))
        self.set_all_body_mass(randomized_body_mass.tolist())

    def randomize_body_inertia(self):
        body_inertia = self._body_inertia_urdf
        randomized_body_inertia = random.uniform(
            np.array(body_inertia) * (1.0 + self.inertia_bound[0]),
            np.array(body_inertia) * (1.0 + self.inertia_bound[1]))
        self.set_all_body_inertia(randomized_body_inertia.tolist())

    def randomize_gravity(self):
        randomized_g = random.uniform(self.g_bound[0], self.g_bound[1])
        self.model.opt.gravity[-1] = - randomized_g
        self.now_g = randomized_g

    def _gait(self, x):
        x = x % (np.pi * 2)
        if np.pi * (3/4) < x < np.pi * (5/4):
            return np.sin(2*(x-np.pi/2))
        elif x <= np.pi * (3/4):
            return np.sin((2/3)*x)
        else:
            return np.sin((2/3)*(x+np.pi))


if __name__ == '__main__':

    laikago = Laikago(visual=True, init_pose=InitPose.LIE, randomized=True)
    laikago.reset()
    t = 0
    T = 2
    while True:
        # print('trunk position' , laikago.sim.data.body_xpos[1])
        t += 1
        action = np.array([[-10, 30, -75],
                           [10, 30, -75],
                           [-10, 50, -75],
                           [10, 50, -75]], dtype=np.float64)
        if t > 60:
            action[[0, 3], 1] += np.sin(t/T)*15
            action[[1, 2], 1] += np.sin(t/T + np.pi)*15
            action[[0, 3], 2] += np.sin(t/T + np.pi/2)*20
            action[[1, 2], 2] += np.sin(t/T + np.pi/2 + np.pi)*20
        action *= (np.pi/180)
        laikago.step(action.flatten())

        if t > 400:
            t = 0
            laikago.reset()

        # print('trunk mass:', laikago.sim.model.body_mass[laikago.model.body_name2id('trunk')])