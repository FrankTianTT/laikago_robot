from builder.tasks.standhigh_task import LaikagoStandHigh2
from builder.gym_env import LaikagoEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import torch

task = LaikagoStandHigh2()
env = LaikagoEnv(task=task, visual=False)
eval_env = LaikagoEnv(task=task, visual=False)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path='logs/',
                             log_path='logs/',
                             eval_freq=10000,
                             deterministic=True,
                             render=False)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
buffer_size = 1000000
batch_size = 64
learning_starts = 10000
ent_coef = 'auto'
time_steps = 5000000

model = SAC('MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./log/",
            policy_kwargs=policy_kwargs,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_starts=learning_starts,
            ent_coef=ent_coef)

if __name__ == "__main__":
    model.learn(total_timesteps=time_steps, callback=eval_callback)