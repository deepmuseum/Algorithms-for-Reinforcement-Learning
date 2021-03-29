"""
Implementation of A2C
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy as c
import pickle
import torch.distributions as torch_dist
import inspect


class ValueNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(0)

    def predict(self, x):
        return self(x).squeeze(0).detach().numpy()


class ActorNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(0)

    def select_action(self, x):
        return torch.multinomial(self.forward(x), 1).cpu().detach().numpy()


class A2CAgent:
    def __init__(
        self,
        env,
        gamma,
        value_network,
        actor_network,
        optimizer,
        device,
        n_a,
        path,
        test_every,
        epochs,
        batch_size,
        timesteps,
        updates,
        entropy_coeff=0.05,
        coeff_loss_value=0.5,
        num_agents=1,
    ):

        self.timesteps = timesteps
        self.updates = updates
        self.epochs = epochs
        self.batch_size = batch_size // num_agents
        self.test_every = test_every
        self.path = path
        self.env = env
        self.gamma = gamma
        self.n_a = n_a
        # Our two networks
        self.value_network = value_network.to(device)
        self.actor_network = actor_network.to(device)
        self.device = device
        # Their optimizer
        self.optimizer = optimizer
        self.best_state_dict = c(self.actor_network.cpu().state_dict())
        self.actor_network.to(self.device)
        self.best_average_reward = -float("inf")
        self.entropy_coeff = entropy_coeff
        self.coeff_loss_value = coeff_loss_value
        self.num_agents = num_agents

    def _returns_advantages(self, rewards, dones, values, next_value):
        """
        """
        # Mnih et al (2016) estimator of the advantage
        targets = torch.zeros((self.timesteps, self.num_agents), device=self.device)
        for i in range(self.timesteps):
            targets[i] = rewards[i]
            if all(dones[i]):
                continue
            for j in range(i + 1, self.timesteps):
                targets[i] += (1 - dones[j]) * self.gamma ** (j - i) * rewards[j]
                if all(dones[j]):
                    break
            if j == self.timesteps - 1 and not all(dones[j]):
                targets[i] += (
                    (1 - dones[j]) * self.gamma ** (self.timesteps - i) * next_value
                )

        advantages = (targets - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # normalization of the advantage estimation from
        # https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
        dones = torch.cat(
            (torch.zeros((1, dones.shape[-1]), device=self.device), dones[1:]), dim=0
        )
        return (1 - dones) * targets, (1 - dones) * advantages, dones

    @staticmethod
    def compute_entropy_bonus(policy):
        return -torch.mean(torch_dist.Categorical(policy).entropy())

    @staticmethod
    def compute_loss_algo(probs, advantages):
        return -torch.mean((torch.log(probs) * advantages))

    @staticmethod
    def compute_loss_value(values, targets):
        return F.mse_loss(values, targets)

    @staticmethod
    def select_from_prob(dist):
        """
        Parameters
        ----------
        dist: torch.Tensor
             representing a prob distribution

        Returns
        -------
        action: int
            index of an action
        """
        action = torch.multinomial(dist, 1).squeeze(-1).tolist()
        return action

    def train(self):
        """
        Perform a training by batch
        """
        episode_count = 0

        observation = self.env.reset()
        tk = tqdm(range(self.updates))
        for update in tk:
            # Lets collect one batch
            actions = []
            dones = []
            rewards = []
            observations = []
            for i in range(self.timesteps):
                if self.num_agents == 1:
                    observation = [observation]
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).unsqueeze(
                    0
                )  # shape (1, num_agents, dim_obs)
                observations.append(observation)
                dist_prob = self.actor_network(
                    observation
                )  # shape (num_agents, num_actions)

                action = self.select_from_prob(dist_prob)  # len num_agents
                actions.append(action)
                if self.num_agents == 1:
                    action = action[0]
                observation, reward, done, info = self.env.step(action)
                if self.num_agents == 1:
                    reward = [reward]
                    done = [done]
                rewards.append(reward)
                dones.append(done)
                if all(dones[-1]):
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state

            next_value = (
                self.value_network(
                    torch.tensor(
                        observation, dtype=torch.float, device=self.device
                    ).unsqueeze(0)
                ).detach()
            ).squeeze(
                -1
            )  # len num_agents
            # Update episode_count

            episode_count += sum(np.all(dones, axis=-1))
            observations = torch.cat(
                observations, dim=0
            )  # shape (bsz, num_agents, dim_obs)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float)
            values = self.value_network(observations).squeeze(-1)
            targets, advantages, dones = self._returns_advantages(
                torch.tensor(rewards, device=self.device), dones, values, next_value
            )  # targets : shape (bsz, num_agents) / advantages : shape (bsz, num_agents)

            actions = torch.tensor(
                actions, device=self.device, dtype=torch.int64
            ).unsqueeze(-1)
            self.optimize_step(observations, actions, targets, advantages, dones)

            # Test it every 50 epochs
            if (update + 1) % self.test_every == 0 or update == self.updates - 1:
                returns_test = np.array(
                    [self.evaluate() for _ in range(100)]  # shape = 100, 4
                )  # shape (100, num_agents)
                mean_returns = np.round(
                    returns_test.mean(axis=0), 2
                )  # shape num_agents
                stds = np.round(returns_test.std(axis=0), 2)
                if np.mean(mean_returns) > self.best_average_reward:
                    self.best_average_reward = np.mean(mean_returns)
                    if self.path is not None:
                        self.best_state_dict = c(self.actor_network.cpu().state_dict())
                        self.actor_network.to(self.device)
                        pickle.dump(self.best_state_dict, open(self.path, "wb"))
                tk.set_postfix(
                    {
                        f"agent {i}": (mean_returns[i], stds[i])
                        for i in range(self.num_agents)
                    }
                )
                observation = self.env.reset()

        print(f"The training was done over a total of {episode_count} episodes")

    def optimize_step(self, observations, actions, targets, advantages, dones):
        index = np.random.permutation(range(self.timesteps))

        for _ in range(self.epochs):
            for j in range(0, self.timesteps, self.batch_size):
                values = (
                    1 - dones[index[j : j + self.batch_size]]
                ) * self.value_network(
                    observations[index[j : j + self.batch_size]]
                ).squeeze(
                    -1
                )  # shape (bsz, num_agents)

                # Compute returns and advantages

                # Learning step !
                self.optimize_minibatch(
                    values,
                    targets[index[j : j + self.batch_size]],
                    advantages[index[j : j + self.batch_size]],
                    actions[index[j : j + self.batch_size]],
                    observations[index[j : j + self.batch_size]],
                )

    def optimize_minibatch(self, values, targets, advantages, actions, observations):
        self.optimizer.zero_grad()
        loss_value = self.compute_loss_value(values, targets)
        distributions = self.actor_network(
            observations
        )  # shape (bsz, num_agents, num_actions)
        probs = distributions.gather(
            -1,
            actions,
            # actions is shape (bsz, num_agents) so unsqueeze -1
        ).squeeze(
            -1
        )  # probs is shape  (bsz, num_agents)

        loss_algo = self.compute_loss_algo(
            probs, advantages
        )  # advantages is shape (bsz, num_agents)
        entropy_bonus = self.compute_entropy_bonus(distributions)
        total_loss = (
            loss_algo
            + self.coeff_loss_value * loss_value
            + self.entropy_coeff * entropy_bonus
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1)
        self.optimizer.step()

    def evaluate(self):
        has_val_param = "val" in inspect.signature(self.env.step).parameters
        observation = self.env.reset()
        observation = torch.tensor(
            observation, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        reward_episode = [0 for _ in range(self.num_agents)]
        done = [False for _ in range(self.num_agents)]

        while not all(done):
            with torch.no_grad():
                policy = self.actor_network(observation)
            action = policy.argmax(dim=-1).tolist()
            if has_val_param:
                observation, reward, done, info = self.env.step(action, val=True)
            else:
                observation, reward, done, info = self.env.step(action)
            observation = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            if self.num_agents == 1:
                reward = [reward]
                done = [done]
            for i, r in enumerate(reward):
                reward_episode[i] += r

        self.env.close()
        return reward_episode


if __name__ == "__main__":
    from torch import optim
    import gym

    environment = gym.make("CartPole-v1")

    value_model = nn.Sequential(
        nn.Linear(environment.observation_space.shape[0], 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

    actor_model = nn.Sequential(
        nn.Linear(environment.observation_space.shape[0], 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, environment.action_space.n),
        nn.Softmax(dim=-1),
    )

    value_network_ = ValueNetwork(value_model)
    actor_network_ = ActorNetwork(actor_model)

    gamma_ = 0.9

    optimizer_ = optim.RMSprop(
        list(value_network_.parameters()) + list(actor_network_.parameters()), lr=0.0001
    )

    device_ = torch.device("cpu")
    agent = A2CAgent(
        env=environment,
        actor_network=actor_network_,
        value_network=value_network_,
        gamma=gamma_,
        optimizer=optimizer_,
        device=device_,
        n_a=environment.action_space.n,
        path=None,
        test_every=10,
        epochs=2,
        batch_size=128,
        timesteps=1000,
        updates=1000,
    )

    agent.train()
