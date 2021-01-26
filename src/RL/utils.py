import torch
import numpy as np
import gym


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


class PseudoEnv:
    def __init__(self, env):
        self.env = env
        self.observation = None
        self.observation_dim = None

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
        return state.astype(np.float).ravel()

    def step(self, action):
        next_observation, reward, done, info = self.env.step(action)
        next_observation = self.preprocess(next_observation)
        delta_observation = (
            next_observation - self.observation
            if self.observation is not None
            else next_observation * 0
        )
        self.observation = next_observation
        return delta_observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self.preprocess(observation)
        self.observation_dim = observation.shape[0]
        return observation

    def close(self):
        self.env.close()
