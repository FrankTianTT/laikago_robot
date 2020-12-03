import importlib
from builder.laikago_env import LaikagoEnv
from gym.wrappers.time_limit import TimeLimit

def build_env(task_name,
              class_name,
              version,
              run_mode,
              simulator,
              visual,
              ctrl_delay,
              action_repeat=20,
              time_step=0.001):
    task_import = importlib.import_module('builder.tasks_' + simulator + '.' + task_name + '_task_' + simulator)
    task = eval('task_import.Laikago' + class_name + simulator.capitalize() +'{}(run_mode="'.format(version) + run_mode + '")')

    env = LaikagoEnv(task=task,
                     visual=visual,
                     ctrl_delay=ctrl_delay,
                     action_repeat=action_repeat,
                     time_step=time_step,
                     simulator=simulator)
    wrapped = TimeLimit(env, task.max_episode_steps)
    return wrapped

if __name__ == '__main__':
    TASK_NAME = 'standup'
    ClASS_NAME = 'StandUp'
    MODE = 'train'
    build_env(TASK_NAME, ClASS_NAME, version=0, run_mode=MODE, simulator='mujoco', visual=True, ctrl_delay=True)