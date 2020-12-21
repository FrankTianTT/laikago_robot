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
                 run_mode='train',
                 init_pose=InitPose.STAND,
                 reward_mode='with_shaping',
                 die_if_unhealthy=False,
                 max_episode_steps=200):
        self._env = None
        self.run_mode = run_mode
        self.init_pose = init_pose
        self.reward_mode = reward_mode
        self.die_if_unhealthy = die_if_unhealthy
        self.max_episode_steps = max_episode_steps
        self.sum_reward = 0
        self.sum_p = 0
        self.steps = 0
        # phi is a function of s, a, s', a' and t
        self.phi_last_state = 0

        self.last_healthy_step = -1
        self.die_after_unhealthy = False

    def reset(self, env):
        self._env = env
        self.steps = 0

    @property
    def is_healthy(self):
        pass

    def update(self):
        self.steps += 1
        if self.is_healthy:
            self.last_healthy_step = self.steps

    def done(self):
        if self.die_after_unhealthy:
            if self.last_healthy_step != -1 and self.last_healthy_step < self.steps + 30:
                return True
            else:
                return False
        elif self.die_if_unhealthy:
            if self.is_healthy:
                return True
            else:
                return False
        else:
            return False

    def add_reward(self, reward, p=1):
        self.sum_reward += reward * p
        self.sum_p += p

    def get_sum_reward(self):
        if self.sum_p == 0:
            reward = 0
        else:
            reward = self.sum_reward / self.sum_p
        self.sum_reward = 0
        self.sum_p = 0
        return reward

    def update_reward(self):
        self.sum_reward = 0
        self.sum_p = 0

    def cal_phi_function(self):
        # 你（可能）需要重载这个函数
        return 0

    @property
    def is_healthy(self):
        # 你（可能）需要重载这个函数
        return True

    def reward(self):
        self.sum_reward = 0
        self.sum_p = 0
        self.update_reward()
        reward = self.get_sum_reward()
        if self.reward_mode == 'with_shaping':
            phi_this_state = self.cal_phi_function()
            shaping_reward = phi_this_state - self.phi_last_state
            self.phi_last_state = phi_this_state
            return reward + shaping_reward
        else:
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