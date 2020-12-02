import torch
import numpy as np
from torch.nn import Sequential, ReLU, Linear, Flatten, Tanh


LOW = np.array([-1.0471976, -0.5235988, -2.7750735, -0.87266463, -0.5235988, -2.7750735,
       -1.0471976, -0.5235988, -2.7750735, -0.87266463, -0.5235988, -2.7750735])
HIGH = np.array([0.87266463, 3.9269907, -0.61086524, 1.0471976, 3.9269907, -0.61086524,
        0.87266463, 3.9269907, -0.61086524, 1.0471976, 3.9269907, -0.61086524])


def load_model():
    model = Sequential(
        Flatten(),
        Linear(138, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 12),
        Tanh()
    )

    params = torch.load('policy.pth', map_location=torch.device('cpu'))

    with torch.no_grad():
        model[1].weight.set_(params['actor.latent_pi.0.weight'])
        model[1].bias.set_(params['actor.latent_pi.0.bias'])
        model[3].weight.set_(params['actor.latent_pi.2.weight'])
        model[3].bias.set_(params['actor.latent_pi.2.bias'])
        model[5].weight.set_(params['actor.mu.weight'])
        model[5].bias.set_(params['actor.mu.bias'])

    return model

def _predict(model, obs):
    return model(obs)

def unscale_action(scaled_action: np.ndarray) -> np.ndarray:
    return LOW + (0.5 * (scaled_action + 1.0) * (HIGH - LOW))

def predict(model, obs, device='cpu'):
    obs = np.array(obs)
    obs = obs.reshape([1, 138])
    obs = torch.as_tensor(obs).float().to(torch.device(device))

    with torch.no_grad():
        actions = _predict(model, obs)

    actions = actions.cpu().numpy()
    actions = unscale_action(actions)

    return actions[0]
