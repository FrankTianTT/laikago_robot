import gym
from transfer.data_transfer import Transfer
from builder.laikago_task import LaikagoTask
import numpy as np
from builder import env_constant
from tasks.task_standup import LaikagoStandUp

class LaikagoEnv(gym.Env):
    def __init__(self,
                 task,
                 visual=True,
                 transfer_class=Transfer):
        self.task = task
        self.task.set_env(self)
        self.visual = visual
        self.transfer_class = transfer_class
        self.transfer = None
        self.transfer = self.transfer_class(visual=self.visual)

    def step(self, action):
        obs = self.transfer.step(action)
        reward = self.task.reward()
        done = self.task.done()
        self.task.update()
        return obs, reward, done, None

    def reset(self):
        self.transfer.reset()
        self.task.reset()
        return self.get_observation()

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

    def get_history_rate_rpy(self):
        return self.transfer.get_history_rate_rpy()

    def get_toe_position(self):
        return self.transfer.get_history_toe_position()

if __name__ == '__main__':
    # task = LaikagoTask()
    task = LaikagoStandUp()
    laikago_env = LaikagoEnv(task=task)

    a = np.array([-15, 15, -35,
                   15, 15, -35,
                   -15, 15, -35,
                   15, 15, -35]) * np.pi / 180
    while True:
        o, r, d, _ = laikago_env.step(a)
        print(r)

        # print('target:', a)
