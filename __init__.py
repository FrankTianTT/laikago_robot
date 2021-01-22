from gym.envs.registration import register
from builder.laikago_env import LaikagoEnv
import builder.tasks_mujoco.runstraight_task_mujoco as running_mujoco
import builder.tasks_mujoco.standup_task_mujoco as standup_mujoco
import builder.tasks_bullet.runstraight_task_bullet as running_bullet
import builder.tasks_bullet.standup_task_bullet as standup_bullet

class LaikagoRunningv0(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = running_mujoco.LaikagoRunStraightMujoco0(run_mode=run_mode)
        super(LaikagoRunningv0, self).__init__(task, visual, simulator='mujoco')

class LaikagoRunningv1(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = running_mujoco.LaikagoRunStraightMujoco1(run_mode=run_mode)
        super(LaikagoRunningv1, self).__init__(task, visual, simulator='mujoco')


class LaikagoStandupv0(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = standup_mujoco.LaikagoStandUpMujoco0(run_mode=run_mode)
        super(LaikagoStandupv0, self).__init__(task, visual, simulator='mujoco')


class LaikagoStandupv1(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = standup_mujoco.LaikagoStandUpMujoco1(run_mode=run_mode)
        super(LaikagoStandupv1, self).__init__(task, visual, simulator='mujoco')

class LaikagoRunningBulletv0(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = running_bullet.LaikagoRunStraightBullet0(run_mode=run_mode)
        super(LaikagoRunningBulletv0, self).__init__(task, visual, simulator='bullet')

class LaikagoRunningBulletv1(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = running_bullet.LaikagoRunStraightBullet1(run_mode=run_mode)
        super(LaikagoRunningBulletv1, self).__init__(task, visual, simulator='bullet')


class LaikagoStandupBulletv0(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = standup_bullet.LaikagoStandUpBullet0(run_mode=run_mode)
        super(LaikagoStandupBulletv0, self).__init__(task, visual, simulator='bullet')


class LaikagoStandupBulletv1(LaikagoEnv):
    def __init__(self,
                 run_mode='train',
                 visual=False):
        task = standup_bullet.LaikagoStandUpBullet1(run_mode=run_mode)
        super(LaikagoStandupBulletv1, self).__init__(task, visual, simulator='bullet')

register(
    id='LaikagoRunning-v0',
    entry_point='LaikagoRunningv0',
)

register(
    id='LaikagoRunning-v1',
    entry_point='LaikagoRunningv1',
)

register(
    id='LaikagoStandup-v0',
    entry_point='LaikagoStandupv0',
)

register(
    id='LaikagoStandup-v1',
    entry_point='LaikagoStandupv1',
)