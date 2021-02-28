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


class ValueNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(0)

    def predict(self, x):
        return self(x).detach().numpy()[0]


class ActorNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).squeeze(0)

    def select_action(self, x, epsilon, num_actions):
        if np.random.uniform() <= epsilon:
            return torch.multinomial(self.forward(x), 1).cpu().detach().numpy()
        else:
            return np.random.choice(range(num_actions))


class A2CAgent:
    def __init__(
        self,
        env,
        gamma,
        value_network,
        actor_network,
        optimizer_value,
        optimizer_actor,
        device,
        obs_dim,
        n_a,
        path,
        test_every,
    ):

        self.test_every = test_every
        self.path = path
        self.epsilon = -1
        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.n_a = n_a
        # Our two networks
        self.value_network = value_network.to(device)
        self.actor_network = actor_network.to(device)
        self.device = device
        # Their optimizers
        self.value_network_optimizer = optimizer_value
        self.actor_network_optimizer = optimizer_actor
        self.best_state_dict = c(self.actor_network.cpu().state_dict())
        self.actor_network.to(self.device)
        self.best_average_reward = -float("inf")

    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network

        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        batch_size = len(rewards)
        returns = np.zeros(batch_size)
        for t in range(batch_size):
            p = t
            while p < batch_size and not dones[p]:
                returns[t] += rewards[p] * self.gamma ** (p - t)
                p = p + 1
            if p == batch_size and not dones[-1]:
                returns[t] += next_value * self.gamma ** (p - t)
        advantages = returns - values
        return returns, advantages

    def select_from_prob(self, dist):
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
        if np.random.uniform() >= self.epsilon:
            action = int(torch.multinomial(dist, 1))
        else:
            action = np.random.choice(range(self.n_a))
        return action

    def training_batch(self, epochs, batch_size):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + self.obs_dim, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []
        tk = tqdm(range(epochs))
        for epoch in tk:
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).unsqueeze(0)
                values[i] = self.value_network(observation)
                self.epsilon = max(0.9 - epoch * 0.001, 0.0)
                dist_prob = self.actor_network(observation)
                action = self.select_from_prob(dist_prob)
                actions[i] = action
                observation, rewards[i], dones[i], info = self.env.step(action)
                if dones[i]:
                    observation = self.env.reset()

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(
                    torch.tensor(
                        observation, dtype=torch.float, device=self.device
                    ).unsqueeze(0)
                )

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(
                rewards, dones, values, next_value
            )

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages)

            # Test it every 50 epochs
            if (epoch + 1) % self.test_every == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(100)]))
                mean_reward = round(rewards_test[-1].mean(), 2)
                if mean_reward > self.best_average_reward:
                    self.best_average_reward = mean_reward
                    self.best_state_dict = c(self.actor_network.cpu().state_dict())
                    self.actor_network.to(self.device)
                    pickle.dump(self.best_state_dict, open(self.path, "wb"))

                tk.set_postfix(
                    {
                        "Mean rewards": mean_reward,
                        "Std": round(rewards_test[-1].std(), 2),
                    }
                )
                observation = self.env.reset()

        print(f"The trainnig was done over a total of {episode_count} episodes")

    def optimize_model(self, observations, actions, returns, advantages):
        returns = torch.tensor(returns[:, None], dtype=torch.float, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        observations = torch.tensor(observations, dtype=torch.float, device=self.device)
        # MSE for the values
        values = self.value_network(observations)
        loss_value = F.mse_loss(values, returns)
        self.value_network_optimizer.zero_grad()
        loss_value.backward()
        # torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1)
        self.value_network_optimizer.step()
        # Actor & Entropy loss
        loss_actor = 0
        for t in range(len(observations)):
            loss_actor -= (
                torch.log(self.actor_network(observations[t].unsqueeze(0))[actions[t]])
                * advantages[t]
                / len(advantages)
            )
        self.actor_network_optimizer.zero_grad()
        loss_actor.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        self.actor_network_optimizer.step()
        return loss_value, loss_actor

    def evaluate(self):
        env = self.env
        observation = env.reset()
        observation = torch.tensor(
            observation, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        reward_episode = 0
        done = False

        while not done:
            with torch.no_grad():
                policy = self.actor_network(observation)
            action = policy.argmax(dim=-1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            reward_episode += reward

        env.close()

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

    gamma_ = 0.99

    value_network_optimizer = optim.RMSprop(value_network_.parameters(), lr=0.001)
    actor_network_optimizer = optim.RMSprop(actor_network_.parameters(), lr=0.001)

    device = torch.device("cuda")
    agent = A2CAgent(
        env=environment,
        actor_network=actor_network_,
        value_network=value_network_,
        gamma=gamma_,
        optimizer_actor=actor_network_optimizer,
        optimizer_value=value_network_optimizer,
        device=device,
        n_a=environment.action_space.n,
    )

    agent.training_batch(1000, 256)
