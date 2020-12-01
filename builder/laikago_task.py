import math
import random
import numpy as np
from enum import Enum
from builder import env_constant
import collections

class InitPose(Enum):
    STAND = 1
    LIE = 2
    ON_ROCK = 3

class LaikagoTask(object):
    def __init__(self,
                 mode='train',
                 init_pose=InitPose.STAND):
        self._env = None
        self.mode = mode
        self.init_pose = init_pose
        self.sum_reward = 0
        self.sum_p = 0
        self.max_episode_steps = 1000
        self.die_if_unhealthy = False
        return

    def reset(self, env):
        pass

    def update(self):
        pass

    def done(self):
        return False

    def reward(self):
        return 0

    def add_reward(self, reward, p=1):
        self.sum_reward += reward * p
        self.sum_p += p

    def get_sum_reward(self):
        reward = self.sum_reward / self.sum_p
        self.sum_reward = 0
        self.sum_p = 0
        return reward

    def normalize_reward(self, reward, min_reward, max_reward):
        return (reward - min_reward)/(max_reward - min_reward)

    def precision_cost(self, v, t, m):
        w = math.atanh(math.sqrt(0.95)) / m
        return math.tanh(((v - t) * w) ** 2)

    def reward_up(self):
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        return 1 - self.precision_cost(math.sqrt(roll ** 2 + pitch ** 2), 0.0, 0.4)

    def reward_still(self):
        chassis_vel = self._env.get_history_chassis_velocity()[0]
        return -math.sqrt(chassis_vel[0] ** 2 + chassis_vel[1] ** 2)

    def reward_rotation(self, r):
        yaw_rate = self._env.get_history_rpy_rate()[0][2]
        k = 1 - self.precision_cost(yaw_rate, 0.0, 0.5)
        return min(k * r, r)

    def reward_turn(self, dir):
        yaw_rate = self._env.get_history_rpy_rate()[0][2]
        return dir * yaw_rate + 0.1 * self.reward_up()

    def reward_lift(self, foot):
        toe_height = []
        for i in range(0, 4):
            toe_height.append(self._env.get_history_toe_position()[0][3*i+2])
        # print(toe_height)
        h = toe_height[foot] - min(toe_height)
        return min(1, h)

    def reward_chassis(self, walk_dir):
        chassis_vel = self._env.get_history_chassis_velocity()[0]
        # print('1', walk_dir)
        # print('2', chassis_vel)
        return self.reward_rotation(np.dot(walk_dir, chassis_vel))

    def toe_swing_velocity(self, foot):
        toe_pos = self._env.get_history_toe_position()[0][3*foot: 3*foot+3]
        last_toe_pos = self._env.get_history_toe_position()[1][3*foot: 3*foot+3]
        v = (toe_pos - last_toe_pos) / env_constant.TIME_STEP
        return np.array(v + self._env.get_history_chassis_velocity()[0])

    def reward_feet(self, walk_dir):
        reward = 0.0
        for i in range(0, 4):
            swing_v = self.toe_swing_velocity(i)
            reward += np.dot(walk_dir, swing_v)
        reward /= 4
        return self.reward_rotation(reward)

    def reward_walk(self, walk_dir):
        return self.reward_chassis(walk_dir) + 0.5 * self.reward_feet(walk_dir) + 0.1 * self.reward_up()

    def reward_toe_contact(self):
        contact = self._env.get_history_toe_collision()[0]
        reward = 1 if sum(contact) == 4 else 0
        return self.normalize_reward(reward, 0, 1)

    def reward_toe_contact_soft(self):
        contact = self._env.get_history_toe_collision()[0]
        reward = sum(contact)
        return self.normalize_reward(reward, -4, 4)

    def reward_min_stand_high(self):
        toe_position = self._env.get_history_toe_position()[0]
        height = []
        for i in [2, 5, 8, 11]:
            height.append(toe_position[i])
        height = - max(height)
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        if height <= 0:
            return height
        else:
            return height * math.cos(roll) * math.cos(pitch)

    def reward_average_stand_high(self):
        toe_position = self._env.get_history_toe_position()[0]
        height = 0
        for i in [2, 5, 8, 11]:
            height += toe_position[i]
        height = - height / 4
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        if height <= 0:
            return height
        else:
            return height * math.cos(roll) * math.cos(pitch)

    def reward_toe_distance(self, threshold=0.2):
        """
        no, x,  y
        0   +   -
        1   +   +
        2   -   -
        3   -   +
        """
        signal = [[1, -1], [1, 1], [-1, -1], [-1, 1]]
        position = self._env.get_history_toe_position()[0]
        x_y_pos = [[position[3 * i], position[3 * i + 1]]for i in range(4)]
        min_distance = min([math.sqrt(position[3 * i] ** 2 + position[3 * i + 1] ** 2) for i in range(4)])
        reward = threshold if min_distance > threshold else min_distance
        for i in range(4):
            if x_y_pos[i][0] * signal[i][0] < 0 or x_y_pos[i][1] * signal[i][1] < 0:
                reward = 0
        return self.normalize_reward(reward, 0, threshold)

    def reward_energy(self):
        energy = self._env.get_energy()
        return - energy

    def done_rp(self, threshold=15):
        r, p, y = self._env.get_history_rpy()[0]
        # print('done rp: ', max(abs(r * 180/np.pi), abs(p * 180/np.pi)))
        return abs(r) > abs(threshold * np.pi / 180) or abs(p) > abs(threshold * np.pi / 180)

    def done_min_stand_high(self, threshold=0.2):
        toe_position = self._env.get_history_toe_position()[0]
        height = [toe_position[i] for i in [2, 5, 8, 11]]
        max_height = - max(height)
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        return max_height * math.cos(roll) * math.cos(pitch) < threshold