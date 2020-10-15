# 这个文件是对laikago的机械特性和电气特性的仿真。

import laikago_constant
from laikago_constant import InitPose
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
import numpy as np
import random

class Laikago(object):
    def __init__(self,
                 visual=False,
                 camera_setting=laikago_constant.CAMERA_SETTING,
                 time_step=laikago_constant.TIME_STEP,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 urdf_filename=laikago_constant.URDF_FILE,
                 init_pose=InitPose.ON_ROCK,
                 self_collision_enabled=False,
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
                 g_bound=laikago_constant.G_BOUND):
        self.visual = visual
        if self.visual:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.camera_setting = camera_setting
        self.time_step = time_step
        self.num_motors = num_motors
        self.num_legs = num_motors / dofs_per_leg
        self._urdf_filename = urdf_filename
        self._init_pose = init_pose
        self._self_collision_enabled = self_collision_enabled
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

        self.now_g = sum(g_bound)/2
        self.last_observation = np.zeros(34).tolist()
        _, self._init_orientation_inv = self._pybullet_client.invertTransform(
            position=[0, 0, 0], orientation=self._get_default_init_orientation())
        self.reset(init_reset=True)
        self.receive_observation()

    def reset(self, init_reset=False):
        if self.visual:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        if init_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setTimeStep(self.time_step)
            self.ground = self._pybullet_client.loadURDF("plane_implicit.urdf")

        self._pybullet_client.resetDebugVisualizerCamera(self.camera_setting['camera_distance'],
                                                         self.camera_setting['camera_yaw'],
                                                         self.camera_setting['camera_pitch'],
                                                         [0, 0, 0])
        if self.visual:
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

        if init_reset:
            self._load_robot_urdf()
            if self._init_pose.value == InitPose.ON_ROCK.value:
                self.rack_constraint = (self._create_rack_constraint(self._get_default_init_position(),
                                                                     self._get_default_init_orientation()))
            self._build_joint_name2Id_dict()
            self._build_urdf_Ids()
            self._remove_default_joint_damping()
            self._build_motor_Id_list()
            self._record_mass_info_from_urdf()
            self._record_inertia_info_from_urdf()
        self._pybullet_client.resetBasePositionAndOrientation(self.quadruped,
                                                              self._get_default_init_position(),
                                                              self._get_default_init_orientation())
        self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        self.reset_pose()
        if self.randomized:
            self.randomize()
        else:
            self._pybullet_client.setGravity(0, 0, - self.now_g)
        return

    def step(self, action):
        """Steps simulation."""
        for i in range(self._action_repeat):
            self._step_internal(action)
        obs = self.last_observation
        self.receive_observation()
        self.last_observation = self.get_observation()
        return obs

    def _step_internal(self, pos_action):
        torque_action = self.position2torque(pos_action)
        self._set_motor_torque_by_Ids(self._motor_id_list, torque_action)
        self._pybullet_client.stepSimulation()
        return

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
        motor_torques = np.clip(motor_torques, -1*self._torque_limits, self._torque_limits)

        return motor_torques

    def _set_motor_torque_by_Ids(self, motor_ids, torques):
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=motor_ids,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            forces=torques)

    def _build_joint_name2Id_dict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _build_urdf_Ids(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._chassis_link_ids = [-1]
        self._hip_link_ids = []
        self._upper_link_ids = []
        self._lower_link_ids = []
        self._toe_link_ids = []

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if laikago_constant.HIP_NAME_PATTERN.match(joint_name):
                self._hip_link_ids.append(joint_id)
            elif laikago_constant.UPPER_NAME_PATTERN.match(joint_name):
                self._upper_link_ids.append(joint_id)
            elif laikago_constant.LOWER_NAME_PATTERN.match(joint_name):
                self._lower_link_ids.append(joint_id)
            elif laikago_constant.TOE_NAME_PATTERN.match(joint_name):
                self._toe_link_ids.append(joint_id)
            else:
                raise ValueError("Unknown category of joint %s" % joint_name)
        self._chassis_link_ids.sort()
        self._hip_link_ids.sort()
        self._upper_link_ids.sort()
        self._lower_link_ids.sort()
        self._toe_link_ids.sort()

        self._leg_link_ids = self._hip_link_ids + self._upper_link_ids + self._lower_link_ids + self._toe_link_ids
        self._movable_joint_ids = self._hip_link_ids + self._upper_link_ids + self._lower_link_ids
        return

    def get_toe_link_ids(self):
        return self._toe_link_ids

    def _remove_default_joint_damping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(
                joint_info[0], -1, linearDamping=0, angularDamping=0)

    def _build_motor_Id_list(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in self._get_motor_names()]

    def _get_motor_names(self):
        return laikago_constant.MOTOR_NAMES

    def _record_mass_info_from_urdf(self):
        """Records the mass information from the URDF file."""
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])

    def get_base_mass_from_urdf(self):
        return self._base_mass_urdf

    def set_base_mass(self, base_mass):
        for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
            self._pybullet_client.changeDynamics(
                self.quadruped, chassis_id, mass=chassis_mass)

    def get_leg_masses_from_urdf(self):
        return self._leg_masses_urdf

    def set_leg_masses(self, leg_masses):
        for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
            self._pybullet_client.changeDynamics(
                self.quadruped, leg_id, mass=leg_mass)

    def _record_inertia_info_from_urdf(self):
        """Record the inertia of each body from URDF file."""
        self._base_inertia_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_inertia_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[2])
        self._leg_inertia_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_inertia_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[2])

    def get_base_inertia_from_urdf(self):
        return self._base_inertia_urdf

    def set_base_inertia(self, base_inertia):
        for chassis_id, chassis_inertia in zip(self._chassis_link_ids, base_inertia):
            self._pybullet_client.changeDynamics(
                self.quadruped, chassis_id, localInertiaDiagonal=chassis_inertia)

    def get_leg_inertias_from_urdf(self):
        return self._leg_inertia_urdf

    def set_leg_inertias(self, leg_inertias):
        for leg_id, leg_inertia in zip(self._chassis_link_ids, leg_inertias):
            self._pybullet_client.changeDynamics(
                self.quadruped, leg_id, localInertiaDiagonal=leg_inertia)

    def get_toe_friction_from_urdf(self):
        toe_frictions = []
        for toe_id in self._toe_link_ids:
            toe_frictions.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, toe_id)[1])
        return toe_frictions

    def set_toe_friction(self, toe_frictions):
        for link_id, toe_friction in zip(self._toe_link_ids, toe_frictions):
            self._pybullet_client.changeDynamics(
                self.quadruped, link_id, lateralFriction=toe_friction)

    def set_joint_friction(self, joint_frictions):
        for joint_id, friction in zip(self._movable_joint_ids, joint_frictions):
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=friction)

    def reset_pose(self):
        if self._init_pose.value == InitPose.LIE.value:
            init_angles = laikago_constant.LIE_MOTOR_ANGLES
        else:
            init_angles = laikago_constant.STAND_MOTOR_ANGLES

        for i, name in enumerate(laikago_constant.MOTOR_NAMES):
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=self._joint_name_to_id[name],
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
            self._pybullet_client.resetJointState(
                self.quadruped, self._joint_name_to_id[name], init_angles[i], targetVelocity=0)

    def _create_rack_constraint(self, init_position, init_orientation):
        """

        :param init_position:
        :param init_orientation:
        :return: the constraint id.
        """
        fixed_constraint = self._pybullet_client.createConstraint(
            parentBodyUniqueId=self.quadruped,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=self._pybullet_client.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=init_position,
            childFrameOrientation=init_orientation)
        return fixed_constraint

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

    def _load_robot_urdf(self):
        urdf_file = self.get_urdf_file()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file,
                self._get_default_init_position(),
                self._get_default_init_orientation(),
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file, self._get_default_init_position(),
                self._get_default_init_orientation())

    def receive_observation(self):
        self._joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self._motor_id_list)
        self._base_position, orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        # Computes the relative orientation relative to the robot_simulation's
        # initial_orientation.
        _, self._base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_orientation_inv)
        angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
        self.roll_pitch_yaw_rate = self.transform_angular_velocity2local_frame(angular_velocity, self._base_orientation)

    def get_true_base_roll_pitch_yaw(self):
        orientation = self._base_orientation
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def get_true_base_roll_pitch_yaw_rate(self):
        return self.roll_pitch_yaw_rate

    def get_true_motor_angles(self):
        motor_angles = [state[0] for state in self._joint_states]
        return motor_angles

    def get_true_motor_velocities(self):
        motor_velocities = [state[1] for state in self._joint_states]
        return motor_velocities

    def _add_sensor_noise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(
            scale=noise_stdev, size=sensor_values.shape)
        return observation

    def transform_angular_velocity2local_frame(self, angular_velocity, orientation):
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                        orientation)
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def get_base_position(self):
        # Attention! This function should be only used in SIMULATION!
        return self._base_position

    def get_base_roll_pitch_yaw(self):
        return self._add_sensor_noise(self.get_true_base_roll_pitch_yaw(),
            self._observation_noise_stdev['IMU_angle'])


    def get_base_roll_pitch_yaw_rate(self):
        return self._add_sensor_noise(self.get_true_base_roll_pitch_yaw_rate(),
            self._observation_noise_stdev['IMU_rate'])

    def get_motor_angles(self):
        return self._add_sensor_noise(np.array(self.get_true_motor_angles()),
            self._observation_noise_stdev['motor_angle'])

    def get_motor_velocities(self):
        return self._add_sensor_noise(
            np.array(self.get_true_motor_velocities()),
            self._observation_noise_stdev['motor_velocity'])

    def get_toe_contacts(self):
        contacts = [-1, -1, -1, -1]
        for i, toe_id in enumerate(self._toe_link_ids):
            contact_points = self._pybullet_client.getContactPoints(bodyA=self.quadruped, bodyB=self.ground, linkIndexA=toe_id)
            if len(contact_points) == 0:
                contacts[i] = -1
            else:
                contacts[i] = 1
        return contacts

    def randomize(self):
        base_mass = self.get_base_mass_from_urdf()
        randomized_base_mass = random.uniform(
            np.array(base_mass) * (1.0 + self.mass_bound[0]),
            np.array(base_mass) * (1.0 + self.mass_bound[1]))
        self.set_base_mass(randomized_base_mass)

        leg_masses = self.get_leg_masses_from_urdf()
        randomized_leg_masses = random.uniform(
            np.array(leg_masses) * (1.0 + self.mass_bound[0]),
            np.array(leg_masses) * (1.0 + self.mass_bound[1]))
        self.set_leg_masses(randomized_leg_masses)

        base_inertial = self.get_base_inertia_from_urdf()
        randomized_base_inertial = random.uniform(
            np.array(base_inertial) * (1.0 + self.mass_bound[0]),
            np.array(base_inertial) * (1.0 + self.mass_bound[1]))
        self.set_base_inertia(randomized_base_inertial)

        leg_inertias = self.get_leg_inertias_from_urdf()
        randomized_leg_inertias = random.uniform(
            np.array(leg_inertias) * (1.0 + self.mass_bound[0]),
            np.array(leg_inertias) * (1.0 + self.mass_bound[1]))
        self.set_leg_inertias(randomized_leg_inertias)

        randomized_toe_friction = random.uniform(
            np.full(4, self.toe_f_bound[0]),
            np.full(4, self.toe_f_bound[1]))
        self.set_toe_friction(randomized_toe_friction)

        randomized_joint_fraction = random.uniform(
            np.full(12, self.joint_f_bound[0]),
            np.full(12, self.joint_f_bound[1]))
        self.set_joint_friction(randomized_joint_fraction)

        randomized_g = random.uniform(self.g_bound[0], self.g_bound[1])
        self._pybullet_client.setGravity(0, 0, - randomized_g)
        self.now_g = randomized_g

    def get_observation(self):
        observation = []
        observation.extend(self.get_motor_angles())                 # [0, 12]
        observation.extend(self.get_motor_velocities())             # [12, 24]
        observation.extend(self.get_base_roll_pitch_yaw())          # [24, 27]
        observation.extend(self.get_base_roll_pitch_yaw_rate())     # [27, 30]
        observation.extend(self.get_toe_contacts())                 # [30, 34]
        return observation

    def print_laikago_info(self):
        print('-'*50)
        print('Information of Laikago as follows.')
        print('mass of chassis: {}'.format(self._pybullet_client.getDynamicsInfo(self.quadruped, -1)[0]))
        print('mass of legs:')
        print([self._pybullet_client.getDynamicsInfo(self.quadruped, id)[0] for id in range(16)])
        print('inertial of chassis: {}'.format(self._pybullet_client.getDynamicsInfo(self.quadruped, -1)[2]))
        print('inertial of legs:')
        print([self._pybullet_client.getDynamicsInfo(self.quadruped, id)[2] for id in range(16)])
        print('g: {}',format(self.now_g))

if __name__ == '__main__':
    laikago = Laikago(visual=False, init_pose=InitPose.STAND)
    laikago.reset()
    action = np.array([-10, 40, -75,
                       10, 40, -75,
                       -10, 40, -75,
                       10, 40, -75]) * np.pi / 180
    # 3 7 11 15
    while True:
        laikago.print_laikago_info()
        laikago.get_toe_contacts()
        laikago.step(action)
        laikago.step(action)
        laikago.reset()

