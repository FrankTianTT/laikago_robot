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
        self.steps = 0
        self.history_rpy_diff = collections.deque(maxlen=100)
        return

    def reset(self, env):
        self._env = env
        self.steps = 0
        self.fall_timer = 0
        self.history_rpy_diff.clear()
        pass

    def update(self):
        self.steps += 1
        history_rpy = self._env.get_history_rpy()
        rpy_diff = ((np.array(history_rpy[0]) - np.array(history_rpy[-1]))**2).sum()
        self.history_rpy_diff.appendleft(rpy_diff)

        roll = self._env.get_history_rpy()[0][0]
        if roll > 3 or roll < -3:
            self.fall_timer += 1
        else:
            self.fall_timer = 0
        pass

    def done(self):
        if self.steps > 1000:
            return True

        if self.fall_timer > 50:
            return True

    def reward(self):
        return 0

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

    def reward_stand_high(self):
        toe_position = self._env.get_history_toe_position()
        height = 0
        for pos in toe_position:
            height += pos[2]
        height = - height/4
        roll = self._env.get_history_rpy()[0][0]
        pitch = self._env.get_history_rpy()[0][1]
        if height <= 0:
            return height
        else:
            return height * math.cos(roll) * math.cos(pitch)

    def reward_energy(self):
        energy = self._env.get_energy()
        return - energy
