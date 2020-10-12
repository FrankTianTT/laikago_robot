import gym
from transfer.data_transfer import Transfer
from builder.laikago_task import LaikagoTask
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pybullet_data as pd
import numpy as np
from builder import env_constant

class LaikagoEnv(gym.Env):
    def __init__(self,
                 task,
                 visual=True,
                 transfer_class=Transfer,
                 camera_setting=env_constant.CAMERA_SETTING):
        self.task = task
        self.visual = visual
        if self.visual:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.ground = self._pybullet_client.loadURDF("plane_implicit.urdf")
        self.transfer = transfer_class(self._pybullet_client)
        self.camera_setting = camera_setting

        self.reset()

    def step(self, action):
        obs = self.transfer.step(action)
        reward = self.task.reward()
        done = self.task.done()
        self.task.update()
        return obs, reward, done, None

    def reset(self):
        self.transfer.reset()
        self._pybullet_client.resetDebugVisualizerCamera(self.camera_setting['camera_distance'],
                                                         self.camera_setting['camera_yaw'],
                                                         self.camera_setting['camera_pitch'],
                                                         [0, 0, 0])
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        self.task.reset()
        return self.get_observation()

    def get_observation(self):
        return self.transfer.get_observation()


if __name__ == '__main__':
    task = LaikagoTask()
    laikago_env = LaikagoEnv(task=task)

    a = np.zeros(12)
    while True:
        laikago_env.transfer.step(a)
