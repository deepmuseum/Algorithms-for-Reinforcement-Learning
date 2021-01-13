"""
Implementation of A2C
"""

import numpy as np
import torch
from RL.utils import make_seed
import torch.nn.functional as F
import torch.nn as nn


# TODO : use GPU


class ValueNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self(x).detach().numpy()[0]


class ActorNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def select_action(self, x):
        return torch.multinomial(self.forward(x), 1).detach().numpy()


class A2CAgent:
    def __init__(
        self,
        env,
        seed,
        gamma,
        value_network,
        actor_network,
        optimizer_value,
        optimizer_actor,
    ):

        self.env = env
        make_seed(seed)
        self.env.seed(seed)
        self.gamma = gamma

        # Our two networks
        self.value_network = value_network
        self.actor_network = actor_network

        # Their optimizers
        self.value_network_optimizer = optimizer_value
        self.actor_network_optimizer = optimizer_actor

    # Hint: use it during training_batch
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
        observations = np.empty(
            (batch_size,) + self.env.observation_space.shape, dtype=np.float
        )
        observation = self.env.reset()
        rewards_test = []

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                values[i] = self.value_network(
                    torch.tensor(observation, dtype=torch.float)
                )
                actions[i] = self.actor_network.select_action(
                    torch.tensor(observation, dtype=torch.float)
                )
                observation, rewards[i], dones[i], info = self.env.step(int(actions[i]))
                if dones[i]:
                    observation = self.env.reset()

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(
                    torch.tensor(observation, dtype=torch.float)
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
            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(
                    f"Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}"
                )

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print("Early stopping !")
                    break
                observation = self.env.reset()

        print(f"The trainnig was done over a total of {episode_count} episodes")

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        observations = torch.tensor(observations, dtype=torch.float)

        # MSE for the values
        values = self.value_network(observations)
        loss_value = F.mse_loss(values, returns)
        self.value_network_optimizer.zero_grad()
        loss_value.backward()
        self.value_network_optimizer.step()
        # Actor & Entropy loss
        loss_actor = 0
        for t in range(len(observations)):
            loss_actor -= (
                torch.log(self.actor_network(observations[t])[int(actions[t][1])])
                * advantages[t]
                / len(advantages)
            )
        self.actor_network_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_network_optimizer.step()
        return loss_value, loss_actor

    def evaluate(self):
        env = self.env
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            policy = self.actor_network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
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

    seed_ = 1
    gamma_ = 0.99

    value_network_optimizer = optim.RMSprop(value_network_.parameters(), lr=0.001)
    actor_network_optimizer = optim.RMSprop(actor_network_.parameters(), lr=0.001)

    agent = A2CAgent(
        env=environment,
        actor_network=actor_network_,
        value_network=value_network_,
        gamma=gamma_,
        seed=seed_,
        optimizer_actor=actor_network_optimizer,
        optimizer_value=value_network_optimizer,
    )

    agent.training_batch(1000, 256)
