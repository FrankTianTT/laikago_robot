from stable_baselines3 import SAC
import argparse
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", default='0', help="Version of task")
    parser.add_argument('-t', '--task_name', default='standup')
    parser.add_argument('-s', '--simulator', default='mujoco')

    args = parser.parse_args()
    version = args.version
    task_name = args.task_name
    simulator = args.simulator

    best_model_save_path = './{}/{}/SAC-v{}/logs/best_model.zip'.format(simulator, task_name, version)
    torch_model_save_path = "./{}_{}_v{}.pt".format(simulator, task_name, version)

    model = SAC.load(best_model_save_path, device=torch.device('cuda:0'))

    torch.save(model.policy.actor.state_dict(), torch_model_save_path, _use_new_zipfile_serialization=False)