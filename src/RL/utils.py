import torch
import numpy as np
from typing import Optional
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gym
from PIL import Image


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


class AtariEnv:
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


class PseudoEnv:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.observation_dim: Optional[tuple] = None
        self.reset()
        self.observation = torch.zeros((4,) + self.observation_dim, device=self.device)

    def get_cart_location(self, screen_width):
        # src:https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # src:https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        resize = T.Compose(
            [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
        )
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode="rgb_array").transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(
                cart_location - view_width // 2, cart_location + view_width // 2
            )
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        output = resize(screen).unsqueeze(0)[:, 0, :, :]
        return output

    def step(self, action):
        done = False
        for _ in range(4):
            next_observation, reward, done, info = self.env.step(action)
            if done:
                break
        next_observation = self.get_screen().squeeze(0)
        tmp = self.observation.clone()
        self.observation[:-1] = tmp[1:]
        self.observation[-1] = next_observation
        return self.observation.unsqueeze(0), reward, done, info

    def reset(self):
        self.env.reset()
        observation = self.get_screen().squeeze(0)
        self.observation_dim = self.get_screen().squeeze(0).shape
        self.observation = torch.zeros((4,) + self.observation_dim, device=self.device)
        tmp = self.observation.clone()
        self.observation[:-1] = tmp[1:]
        self.observation[-1] = observation
        return self.observation.unsqueeze(0)

    def close(self):
        self.env.close()


if __name__ == "__main__":

    env_ = gym.make("CartPole-v0").unwrapped
    env = PseudoEnv(env_, "cpu")
    env.reset()
    im = env.get_screen().squeeze(0)
    plt.imshow(im)
    plt.show()
