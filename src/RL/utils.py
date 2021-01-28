import torch
import numpy as np
from typing import Optional


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


class PseudoEnv:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.observation_dim: Optional[tuple] = None
        self.reset()
        self.observation = torch.zeros((4,) + self.observation_dim, device=self.device)

    def preprocess(self, state):
        """
        preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
        1. crop
        2. downsample by factor of 2
        3. erase background (background type 1)
        4. erase background (background type 2)
        5. everything else (paddles, ball) just set to 1
        """

        state = state[35:195]
        state = state[::2, ::2, 0]
        state[state == 144] = 0
        state[state == 109] = 0
        state[state != 0] = 1
        return state

    def step(self, action):
        done = False
        for _ in range(4):
            next_observation, reward, done, info = self.env.step(action)
            if done:
                break

        next_observation = torch.tensor(
            next_observation, dtype=torch.float, device=self.device
        )
        next_observation = self.preprocess(next_observation)

        # delta_observation = (
        #     next_observation - self.observation
        #     if self.observation is not None
        #     else next_observation
        # )

        tmp = self.observation.clone()
        self.observation[:-1] = tmp[1:]
        self.observation[-1] = next_observation
        return self.observation.unsqueeze(0), reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = torch.tensor(observation, dtype=torch.float, device=self.device)
        observation = self.preprocess(observation)
        self.observation_dim = observation.shape
        self.observation = torch.zeros((4,) + self.observation_dim, device=self.device)
        tmp = self.observation.clone()
        self.observation[:-1] = tmp[1:]
        self.observation[-1] = observation
        return self.observation.unsqueeze(0)

    def close(self):
        self.env.close()
