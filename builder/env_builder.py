import importlib
from builder.laikago_env import LaikagoEnv
from gym.wrappers.time_limit import TimeLimit
from gym.wrappers import Monitor

def str2Hump(text):
    str_list = text.split('_')
    hump = ""
    for s in str_list:
        hump += s.capitalize()
    return hump

def build_env(task_name,
              class_name,
              version,
              run_mode,
              simulator,
              visual,
              ctrl_delay,
              action_repeat=20,
              time_step=0.001,
              record_dir=None):
    task_import = importlib.import_module('builder.tasks_' + simulator + '.' + task_name + '_task_' + simulator)
    task = eval('task_import.Laikago' + class_name + str2Hump(simulator) +'{}(run_mode="'.format(version) + run_mode + '")')
    env = LaikagoEnv(task=task,
                     visual=visual,
                     ctrl_delay=ctrl_delay,
                     action_repeat=action_repeat,
                     time_step=time_step,
                     simulator=simulator)
    if record_dir is not None:
        env = Monitor(env, record_dir)
    env = TimeLimit(env, task.max_episode_steps)
    return env

if __name__ == '__main__':
    TASK_NAME = 'standup'
    ClASS_NAME = 'StandUp'
    RUN_MODE = 'report_done'
    SIMULATOR = 'mujoco_torque'

    build_env(TASK_NAME, ClASS_NAME, version=0, run_mode=RUN_MODE, simulator=SIMULATOR, visual=True, ctrl_delay=True)