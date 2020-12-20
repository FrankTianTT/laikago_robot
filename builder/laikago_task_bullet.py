import math
import random
import inspect
import sys
import numpy as np
from enum import Enum
from builder import env_constant
import collections
from builder.laikago_task import LaikagoTask

# Attention! This class use some function that not provided in REALITY.

class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

class LaikagoTaskBullet(LaikagoTask):
    def __init__(self,
                 run_mode='train',
                 reward_mode='with_shaping',
                 die_if_unhealthy=False,
                 max_episode_steps=1000,
                 init_pose=InitPose.STAND,
                 contact_buffer_length=5):
        super(LaikagoTaskBullet, self).__init__(run_mode=run_mode,
                                                reward_mode=reward_mode,
                                                die_if_unhealthy=die_if_unhealthy,
                                                max_episode_steps=max_episode_steps,
                                                init_pose=init_pose)
        self.contact_buffer_length = contact_buffer_length
        self.contact_buffer = collections.deque(maxlen=self.contact_buffer_length)
        for i in range(self.contact_buffer_length):
            self.contact_buffer.appendleft(4)

    def update(self):
        super(LaikagoTaskBullet, self).update()
        contact = (sum(self._env.get_history_toe_collision()[0]) + 4) / 2
        self.contact_buffer.appendleft(contact)
        if self.run_mode is "report_done":
            print(self.steps)

    def reward_energy(self, threshold=0.2):
        energy = self._env.get_energy()
        reward = 1 if energy < threshold else threshold / energy
        return reward

    def done_toe_height_bullet(self, threshold=0.03):
        max_height = max(self._env.transfer.laikago.get_toe_height_for_reward())
        return max_height > threshold

    def reward_toe_height_bullet(self, threshold=0.03):
        max_height = max(self._env.transfer.laikago.get_toe_height_for_reward())
        return 1 if max_height < threshold else threshold / max_height

    def done_r_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        done = abs(r) > abs(threshold * np.pi / 180)
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_r_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        threshold = threshold * np.pi / 180
        reward = 1 if r < threshold else threshold / r
        return reward

    def done_p_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        done = abs(p) > abs(threshold * np.pi / 180)
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_p_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        threshold = threshold * np.pi / 180
        reward = 1 if p < threshold else threshold / p
        return reward

    def done_y_bullet(self, threshold=30):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        done = abs(y) > abs(threshold * np.pi / 180)
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_y_bullet(self, threshold=15):
        r, p, y = self._env.transfer.laikago.get_rpy_for_reward()
        threshold = threshold * np.pi / 180
        reward = 1 if y < threshold else threshold / y
        return reward

    def done_height_bullet(self, threshold=0.35):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        height = base_pos[2]
        # print(height)
        done = height < threshold
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_height_bullet(self, threshold=0.3):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        h = base_pos[2]
        reward = 1 if h > threshold else h / threshold
        return reward

    def done_x_velocity(self, threshold=1):
        x_vel = self._env.transfer.laikago.get_velocity_for_reward()[0]
        done = x_vel < threshold
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_x_velocity(self, threshold=1):
        x_vel = self._env.transfer.laikago.get_velocity_for_reward()[0]
        reward = 1 if x_vel > threshold else x_vel / threshold
        return reward

    def done_y_velocity(self, threshold=0.1):
        y_vel = self._env.transfer.laikago.get_velocity_for_reward()[1]
        done = y_vel > threshold
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_y_velocity(self, threshold=0.1):
        y_vel = self._env.transfer.laikago.get_velocity_for_reward()[0]
        reward = 1 if y_vel < threshold else threshold / y_vel
        return reward

    def done_region_bullet(self, threshold=0.5):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        x = base_pos[0] ** 2 + base_pos[1] ** 2
        done = x > threshold ** 2
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_region_bullet(self, threshold=1):
        base_pos = self._env.transfer.laikago.get_position_for_reward()
        x = base_pos[0] ** 2 + base_pos[1] ** 2
        reward = 1 if x < threshold else threshold/x
        return reward

    def done_toe_contact(self, threshold=5):
        if self.steps < threshold:
            return False
        contact = self._env.get_history_toe_collision()[0]
        done = sum(contact) != 4
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_toe_contact(self):
        contact = self._env.get_history_toe_collision()[0]
        reward = 1 if sum(contact) == 4 else 0
        return self.normalize_reward(reward, 0, 1)

    def done_toe_contact_long(self, threshold=15):
        done = sum(self.contact_buffer) < threshold
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def done_toe_distance(self, threshold=0.2):
        """
        no, x,  y
        0   +   -
        1   +   +
        2   -   -
        3   -   +
        """
        signal = [[1, -1], [1, 1], [-1, -1], [-1, 1]]
        position = self._env.get_history_toe_position()[0]
        x_y_pos = [[position[3 * i], position[3 * i + 1]] for i in range(4)]
        min_distance = min([math.sqrt(position[3 * i] ** 2 + position[3 * i + 1] ** 2) for i in range(4)])
        for i in range(4):
            if x_y_pos[i][0] * signal[i][0] < 0 or x_y_pos[i][1] * signal[i][1] < 0:
                return True
        done = min_distance < threshold
        if done and self.run_mode is "report_done":
            print(self.get_function_name())
        return done

    def reward_toe_distance(self, threshold=0.2):
        signal = [[1, -1], [1, 1], [-1, -1], [-1, 1]]
        position = self._env.get_history_toe_position()[0]
        x_y_pos = [[position[3 * i], position[3 * i + 1]] for i in range(4)]
        min_distance = min([math.sqrt(position[3 * i] ** 2 + position[3 * i + 1] ** 2) for i in range(4)])
        reward = 1 if min_distance < threshold else threshold / min_distance
        for i in range(4):
            if x_y_pos[i][0] * signal[i][0] < 0 or x_y_pos[i][1] * signal[i][1] < 0:
                reward = 0
        reward = 1 if min_distance > threshold else min_distance / threshold
        return reward

    def reward_toe_contact_long(self, threshold=15):
        return 1 if sum(self.contact_buffer) > threshold else sum(self.contact_buffer) / threshold

    @staticmethod
    def get_function_name():
        return inspect.stack()[1][3]