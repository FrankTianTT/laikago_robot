import gym
from gym import spaces
from transfer.data_transfer import Transfer
from builder.laikago_task import LaikagoTask
import time
import numpy as np
from builder import env_constant
from tasks.task_standup import LaikagoStandUp

class LaikagoEnv(gym.Env):
    def __init__(self,
                 task,
                 visual=True,
                 transfer_class=Transfer,
                 position_upper_bound=env_constant.POSITION_UPPER_BOUND,
                 position_lower_bound=env_constant.POSITION_LOWER_BOUND):
        self.task = task
        self.task.set_env(self)
        self.visual = visual
        self.transfer_class = transfer_class
        self.transfer = None
        self.transfer = self.transfer_class(visual=self.visual, init_pose=self.task.init_pose)
        self.action_space = spaces.Box(
            np.array(position_upper_bound),
            np.array(position_lower_bound),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            np.ones(46*3),
            - np.ones(46*3),
            dtype=np.float32)


    def step(self, action):
        transfer_obs = self.transfer.step(action)
        obs = []
        for o in transfer_obs:
            obs.extend(o)
        reward = self.task.reward()
        done = self.task.done()
        self.task.update()
        return obs, reward, done, None

    def reset(self):
        transfer_obs = self.transfer.reset()
        obs = []
        for o in transfer_obs:
            obs.extend(o)
        self.task.reset()
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
    # task = LaikagoTask()
    task = LaikagoStandUp()
    laikago_env = LaikagoEnv(task=task, visual=False)

    a = np.array([-15, 15, -35,
                   15, 15, -35,
                   -15, 15, -35,
                   15, 15, -35]) * np.pi / 180
    while True:
        o, r, d, _ = laikago_env.step(a)
        print(len(o))

        # print('target:', a)