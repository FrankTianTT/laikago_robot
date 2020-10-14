import gym
from gym import spaces
from transfer.data_transfer import Transfer
from builder.laikago_task import LaikagoTask
import time
import numpy as np
from builder import env_constant
from tasks.standupright_task import LaikagoStandUpright
from tasks.turn_task import LaikagoTurn
from tasks.liftfoot_task import LaikagoLiftFoot
from tasks.walk_task import LaikagoWalk

class LaikagoEnv(gym.Env):
    def __init__(self,
                 task,
                 visual=True,
                 transfer_class=Transfer,
                 position_upper_bound=env_constant.POSITION_UPPER_BOUND,
                 position_lower_bound=env_constant.POSITION_LOWER_BOUND):
        self.task = task
        self.visual = visual
        self.transfer_class = transfer_class
        self.transfer = None
        self.transfer = self.transfer_class(visual=self.visual, init_pose=self.task.init_pose)
        self.action_space = spaces.Box(
            np.array(position_upper_bound),
            np.array(position_lower_bound),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            np.ones(90),
            - np.ones(90),
            dtype=np.float32)


    def step(self, action):
        obs = self.transfer.step(action)
        self.task.update()

        done = self.task.done()
        reward = self.task.reward()
        return obs, reward, done, {}

    def reset(self):
        obs = self.transfer.reset()
        self.task.reset(self)
        return obs

    def get_observation(self):
        return self.transfer.get_observation()

    def get_history_velocity(self):
        return self.transfer.get_history_velocity()

    def get_history_chassis_velocity(self):
        return self.transfer.get_history_chassis_velocity()

    def get_history_angle(self):
        return self.transfer.get_history_angle()

    def get_history_rpy(self):
        return self.transfer.get_history_rpy()

    def get_history_rpy_rate(self):
        return self.transfer.get_history_rpy_rate()

    def get_history_toe_position(self):
        return self.transfer.get_history_toe_position()

if __name__ == '__main__':
    task = LaikagoWalk(dir=env_constant.WALK_RIGHT)
    laikago_env = LaikagoEnv(task=task, visual=False)

    a = np.array([-50, 15, -35,
                   50, 15, -35,
                   -50, 15, -35,
                   50, 15, -35]) * np.pi / 180
    while True:
        o, r, d, _ = laikago_env.step(a)
        # print(laikago_env.transfer.get_chassis_vel_by_toe())
