# 这个文件是对laikago的机械特性和电气特性的仿真。

import laikago_constant
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
from laikago_config import InitPose
from laiakgo_randomizer import LaikagoRobotRandomizer
import numpy as np
import collections

class Laikago(object):
    def __init__(self,
                 pybullet_client,
                 time_step=0.01,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 urdf_filename=laikago_constant.URDF_FILE,
                 init_pose=InitPose.STAND,
                 self_collision_enabled=False,
                 action_repeat=1,
                 randomized=True,
                 observation_noise_stdev=laikago_constant.SENSOR_NOISE_STDDEV,
                 control_latency=0.0):
        self._pybullet_client = pybullet_client
        self.time_step = time_step
        self.num_motors = num_motors
        self.num_legs = num_motors / dofs_per_leg
        self._urdf_filename = urdf_filename
        self._init_pose = init_pose
        self._self_collision_enabled = self_collision_enabled
        self._action_repeat = action_repeat
        self.randomized = randomized
        self._observation_noise_stdev = observation_noise_stdev
        self._control_latency = control_latency
        if self.randomized:
            self.randomizer = LaikagoRobotRandomizer(self)

        _, self._init_orientation_inv = self._pybullet_client.invertTransform(
            position=[0, 0, 0], orientation=self._GetDefaultInitOrientation())
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []

        self.Reset()
        self.ReceiveObservation()

    def Reset(self):
        self._LoadRobotURDF()
        if self._init_pose == InitPose.ON_POCK:
            self.rack_constraint = (self._CreateRackConstraint(self._GetDefaultInitPosition(),
                                                               self._GetDefaultInitOrientation()))
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._RemoveDefaultJointDamping()
        self._BuildMotorIdList()
        self._RecordMassInfoFromURDF()
        self._RecordInertiaInfoFromURDF()
        self._pybullet_client.resetBasePositionAndOrientation(self.quadruped,
                                                              self._GetDefaultInitPosition(),
                                                              self._GetDefaultInitOrientation())
        self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        self.ResetPose()
        if self.randomized:
            self.randomizer.randomize()
        return

    def Step(self, action):
        """Steps simulation."""
        for i in range(self._action_repeat):
            self._StepInternal(action)
        return

    def _StepInternal(self, action):
        self._SetMotorTorqueByIds(self._motor_id_list, action)
        self._pybullet_client.stepSimulation()
        self.ReceiveObservation()
        return

    def _SetMotorTorqueByIds(self, motor_ids, torques):
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=motor_ids,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            forces=torques)

    def ReceiveObservation(self):
        self._joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self._motor_id_list)
        self._base_position, orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
        # Computes the relative orientation relative to the robot's
        # initial_orientation.
        _, self._base_orientation = self._pybullet_client.multiplyTransforms(
            positionA=[0, 0, 0],
            orientationA=orientation,
            positionB=[0, 0, 0],
            orientationB=self._init_orientation_inv)
        self._observation_history.appendleft(self.GetTrueObservation())
        self._control_observation = self._GetControlObservation()

    def _GetControlObservation(self):
        control_delayed_observation = self._GetDelayedObservation(
            self._control_latency)
        return control_delayed_observation

    def _BuildJointNameToIdDict(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
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

    def _RemoveDefaultJointDamping(self):
        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
            self._pybullet_client.changeDynamics(
                joint_info[0], -1, linearDamping=0, angularDamping=0)

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in self._GetMotorNames()]

    def _GetMotorNames(self):
        return laikago_constant.MOTOR_NAMES

    def _RecordMassInfoFromURDF(self):
        """Records the mass information from the URDF file."""
        self._base_mass_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
        self._leg_masses_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_masses_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])

    def GetBaseMassFromURDF(self):
        return self._base_mass_urdf

    def SetBaseMass(self, base_mass):
        for chassis_id, chassis_mass in zip(self._chassis_link_ids, base_mass):
            self._pybullet_client.changeDynamics(
                self.quadruped, chassis_id, mass=chassis_mass)

    def GetLegMassesFromURDF(self):
        return self._leg_masses_urdf

    def SetLegMasses(self, leg_masses):
        for leg_id, leg_mass in zip(self._leg_link_ids, leg_masses):
            self._pybullet_client.changeDynamics(
                self.quadruped, leg_id, mass=leg_mass)

    def _RecordInertiaInfoFromURDF(self):
        """Record the inertia of each body from URDF file."""
        self._base_inertia_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_inertia_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[2])
        self._leg_inertia_urdf = []
        for leg_id in self._leg_link_ids:
            self._leg_inertia_urdf.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[2])

    def GetBaseInertiaFromURDF(self):
        return self._base_inertia_urdf

    def SetBaseInertia(self, base_inertia):
        for chassis_id, chassis_inertia in zip(self._chassis_link_ids, base_inertia):
            self._pybullet_client.changeDynamics(
                self.quadruped, chassis_id, localInertiaDiagonal=chassis_inertia)

    def GetLegInertiasFromURDF(self):
        return self._leg_inertia_urdf

    def SetLegInertias(self, leg_inertias):
        for leg_id, leg_inertia in zip(self._chassis_link_ids, leg_inertias):
            self._pybullet_client.changeDynamics(
                self.quadruped, leg_id, localInertiaDiagonal=leg_inertia)

    def GetToeFrictionFromURDF(self):
        toe_frictions = []
        for toe_id in self._toe_link_ids:
            toe_frictions.append(
                self._pybullet_client.getDynamicsInfo(self.quadruped, toe_id)[1])
        return toe_frictions

    def SetToeFriction(self, toe_frictions):
        for link_id, toe_friction in zip(self._toe_link_ids, toe_frictions):
            self._pybullet_client.changeDynamics(
                self.quadruped, link_id, lateralFriction=toe_friction)

    def SetJointFriction(self, joint_frictions):
        for joint_id, friction in zip(self._movable_joint_ids, joint_frictions):
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=friction)

    def ResetPose(self):
        if self._init_pose == InitPose.LIE:
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

    def _CreateRackConstraint(self, init_position, init_orientation):
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

    def GetURDFFile(self):
        return self._urdf_filename

    def _GetDefaultInitPosition(self):
        if self._init_pose == InitPose.ON_POCK:
            return laikago_constant.ON_RACK_INIT_POSITION
        elif self._init_pose == InitPose.STAND:
            return laikago_constant.STAND_INIT_POSITION
        else:
            return laikago_constant.LIE_INIT_POSITION

    def _GetDefaultInitOrientation(self):
        return laikago_constant.INIT_ORIENTATION

    def _LoadRobotURDF(self):
        urdf_file = self.GetURDFFile()
        if self._self_collision_enabled:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file,
                self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation(),
                flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
        else:
            self.quadruped = self._pybullet_client.loadURDF(
                urdf_file, self._GetDefaultInitPosition(),
                self._GetDefaultInitOrientation())

    def _AddSensorNoise(self, sensor_values, noise_stdev):
        if noise_stdev <= 0:
            return sensor_values
        observation = sensor_values + np.random.normal(
            scale=noise_stdev, size=sensor_values.shape)
        return observation

    def _GetDelayedObservation(self, latency):
        if latency <= 0 or len(self._observation_history) == 1:
            observation = self._observation_history[0]
        else:
            n_steps_ago = int(latency / self.time_step)
            if n_steps_ago + 1 >= len(self._observation_history):
                return self._observation_history[-1]
            remaining_latency = latency - n_steps_ago * self.time_step
            blend_alpha = remaining_latency / self.time_step
            observation = (
                    (1.0 - blend_alpha) * np.array(self._observation_history[n_steps_ago])
                    + blend_alpha * np.array(self._observation_history[n_steps_ago + 1]))
        return observation

    def GetTrueBaseOrientation(self):
        return self._base_orientation

    def GetTrueBaseRollPitchYaw(self):
        orientation = self.GetTrueBaseOrientation()
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
        return np.asarray(roll_pitch_yaw)

    def GetBaseRollPitchYaw(self):
        return self._AddSensorNoise(np.array(
            self._control_observation[2 * self.num_motors: 2 * self.num_motors + 3]),
            self._observation_noise_stdev['IMU_angle'])

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                        orientation)
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def GetTrueBaseRollPitchYawRate(self):
        angular_velocity = self._pybullet_client.getBaseVelocity(self.quadruped)[1]
        orientation = self.GetTrueBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)

    def GetBaseRollPitchYawRate(self):
        return self._AddSensorNoise(
            np.array(self._control_observation[2 * self.num_motors + 3: 2 * self.num_motors + 6]),
            self._observation_noise_stdev['IMU_rate'])

    def GetTrueMotorAngles(self):
        motor_angles = [state[0] for state in self._joint_states]
        return motor_angles

    def GetMotorAngles(self):
        return self._AddSensorNoise(
            np.array(self._control_observation[0:self.num_motors]),
            self._observation_noise_stdev['motor_angle'])

    def GetTrueMotorVelocities(self):
        motor_velocities = [state[1] for state in self._joint_states]
        return motor_velocities

    def GetMotorVelocities(self):
        return self._AddSensorNoise(
            np.array(self._control_observation[self.num_motors:2 * self.num_motors]),
            self._observation_noise_stdev['motor_velocity'])

    def GetToeContacts(self):
        contact_ids = [i[3] for i in self._pybullet_client.getContactPoints(bodyA=self.quadruped)]
        toe_contacts = [0, 0, 0, 0]
        for i, toe_id in enumerate(self._toe_link_ids):
            if toe_id in contact_ids:
                toe_contacts[i] = 1
        return toe_contacts

    def GetTrueObservation(self):
        observation = []
        observation.extend(self.GetTrueMotorAngles())           # [0, 12]
        observation.extend(self.GetTrueMotorVelocities())       # [12, 24]
        observation.extend(self.GetTrueBaseRollPitchYaw())      # [24, 27]
        observation.extend(self.GetTrueBaseRollPitchYawRate())  # [27, 30]
        observation.extend(self.GetToeContacts())               # [30, 33]
        return observation

    def GetObservation(self):
        observation = []
        observation.extend(self.GetMotorAngles())               # [0, 12]
        observation.extend(self.GetMotorVelocities())           # [12, 24]
        observation.extend(self.GetBaseRollPitchYaw())          # [24, 27]
        observation.extend(self.GetBaseRollPitchYawRate())      # [27, 30]
        observation.extend(self.GetToeContacts())               # [30, 33]

    def PrintLaikagoInfo(self):
        print('-'*50)
        print('Information of Laikago as follows.')
        print('Name of Motor:')
        print(self._GetMotorNames())
        print('mass of chassis:')
        print(self._base_mass_urdf)
        print('mass of legs:')
        print(self._leg_masses_urdf)
        print('inertial of chassis:')
        print(self._base_inertia_urdf)
        print('inertial of legs:')
        print(self._leg_inertia_urdf)

if __name__ == '__main__':
    pyb = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    pyb.setGravity(0, 0, -10)
    pyb.setAdditionalSearchPath(pd.getDataPath())
    ground = pyb.loadURDF("plane_implicit.urdf")
    laikago = Laikago(pyb,
                      init_pose=InitPose.LIE)
    laikago.PrintLaikagoInfo()

    while True:
        laikago.Reset()