import gym
from gym import spaces
from transfer.data_transfer import Transfer
from builder.laikago_task import LaikagoTask
import time
import numpy as np
from builder import env_constant

class LaikagoEnv(gym.Env):
    def __init__(self,
                 task,
                 visual=True,
                 transfer_class=Transfer,
                 ctrl_delay=False,
                 action_repeat=33,
                 position_upper_bound=env_constant.POSITION_UPPER_BOUND,
                 position_lower_bound=env_constant.POSITION_LOWER_BOUND):
        self.task = task
        self.task.reset(self)
        self.visual = visual
        self.transfer_class = transfer_class
        self.ctrl_delay = ctrl_delay
        self.action_repeat = action_repeat
        self.transfer = None
        self.transfer = self.transfer_class(visual=self.visual,
                                            init_pose=self.task.init_pose,
                                            action_repeat=self.action_repeat,
                                            ctrl_delay=self.ctrl_delay)
        self.action_space = spaces.Box(
            np.array(position_lower_bound, dtype=np.float32),
            np.array(position_upper_bound, dtype=np.float32))
        obs_size = 46 * 3
        self.observation_space = spaces.Box(
            np.ones(obs_size),
            - np.ones(obs_size),
            dtype=np.float32)
        self.energy = 0


    def step(self, action):
        obs, energy = self.transfer.step(action)
        self.energy = energy
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

    def get_history_toe_collision(self):
        return self.transfer.get_history_toe_collision()

    def get_energy(self):
        return self.energy

if __name__ == '__main__':
    from builder.tasks_bullet.standup_task_bullet import LaikagoStandUpBullet7_1 as Task
    task = Task()
    laikago_env = LaikagoEnv(task=task, visual=False,
                             obs_delay=True)

    a = np.array([-50, 15, -35,
                   50, 15, -35,
                   -50, 15, -35,
                   50, 15, -35]) * np.pi / 180

    print(len(laikago_env.observation_space.sample()))
    # while True:
    #     o, r, d, _ = laikago_env.step(a)
    #     # print(laikago_env.transfer.get_chassis_vel_by_toe())
