# 这个文件是对laikago的机械特性和电气特性的仿真。

import laikago_constant
import pybullet
import pybullet_utils.bullet_client as bullet_client
from laikago_config import InitPose

class Laikago(object):
    def __init__(self,
                 pybullet_client,
                 num_motors=laikago_constant.NUM_MOTORS,
                 dofs_per_leg=laikago_constant.DOFS_PER_LEG,
                 urdf_filename=laikago_constant.URDF_FILE,
                 init_pose=InitPose.STAND,
                 self_collision_enabled=False):
        self._pybullet_client = pybullet_client
        self._urdf_filename = urdf_filename
        self._init_pose = init_pose
        self._self_collision_enabled = self_collision_enabled

        self.Reset()

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
        self._pybullet_client.resetBaseVelocity(self.quadruped,
                                                [0, 0, 0],
                                                [0, 0, 0])
        self.ResetPose(add_constraint=True)
    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
          ValueError: Unknown category of the joint name.
        """
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
        return

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


if __name__ == '__main__':
    pyb = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    laikago = Laikago(pyb)