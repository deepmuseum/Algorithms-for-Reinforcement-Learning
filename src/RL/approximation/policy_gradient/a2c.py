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

# TODO : multi-agent training


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
        optimizer,
        device,
        n_a,
        path,
        test_every,
        epochs,
        batch_size,
        entropy_coeff=0.1,
        coeff_loss_value=0.1,
    ):

        self.epochs = epochs
        self.batch_size = batch_size
        self.test_every = test_every
        self.path = path
        self.epsilon = -1
        self.env = env
        self.gamma = gamma
        self.n_a = n_a
        # Our two networks
        self.value_network = value_network.to(device)
        self.actor_network = actor_network.to(device)
        self.device = device
        # Their optimizers
        self.optimizer = optimizer
        self.best_state_dict = c(self.actor_network.cpu().state_dict())
        self.actor_network.to(self.device)
        self.best_average_reward = -float("inf")
        self.entropy_coeff = entropy_coeff
        self.coeff_loss_value = coeff_loss_value

    def _returns_advantages(self, rewards, dones, values, next_value):
        """
        """
        # Mnih et al (2016) estimator of the advantage
        targets = torch.zeros(self.batch_size, device=self.device)
        i = 0
        for i in range(self.batch_size):
            targets[i] = rewards[i]
            if dones[i]:
                continue
            for j in range(i + 1, self.batch_size):
                targets[i] += self.gamma ** (j - i) * rewards[j]
                if dones[j]:
                    break
            if j == self.batch_size - 1 and not dones[j]:
                targets[i] += self.gamma ** (j + 1 - i) * next_value

        advantages = (targets - values).detach()

        return targets, advantages

    @staticmethod
    def compute_entropy_bonus(policy):
        return -torch.sum(torch_dist.Categorical(policy).entropy())

    @staticmethod
    def compute_loss_algo(probs, advantages):
        return -torch.sum((torch.log(probs) * advantages / len(advantages)))

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
        action = int(torch.multinomial(dist, 1))
        return action

    def train(self):
        """
        Perform a training by batch
        """
        episode_count = 0

        observation = self.env.reset()
        rewards_test = []
        tk = tqdm(range(self.epochs))
        for epoch in tk:
            # Lets collect one batch
            actions = []
            dones = []
            rewards = []
            observations = []
            for i in range(self.batch_size):
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).unsqueeze(0)
                observations.append(observation)
                dist_prob = self.actor_network(observation)
                action = self.select_from_prob(dist_prob)
                actions.append(action)
                observation, reward, done, info = self.env.step(action)
                rewards.append(reward)
                dones.append(done)
                if dones[-1]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0

            else:
                next_value = self.value_network(
                    torch.tensor(
                        observation, dtype=torch.float, device=self.device
                    ).unsqueeze(0)
                ).item()

            # Update episode_count
            episode_count += sum(dones)
            observations = torch.cat(observations, dim=0)
            values = self.value_network(observations).squeeze(-1)

            # Compute returns and advantages
            targets, advantages = self._returns_advantages(
                torch.tensor(rewards, device=self.device),
                torch.tensor(dones, device=self.device),
                values,
                next_value,
            )

            # Learning step !
            self.optimize_model(values, targets, advantages, actions, observations)

            # Test it every 50 epochs
            if (epoch + 1) % self.test_every == 0 or epoch == self.epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(100)]))
                mean_reward = round(rewards_test[-1].mean(), 2)
                if mean_reward > self.best_average_reward:
                    self.best_average_reward = mean_reward
                    if self.path is not None:
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

    def optimize_model(self, values, targets, advantages, actions, observations):
        loss_value = self.compute_loss_value(values, targets)
        # torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1)
        distributions = self.actor_network(observations)
        probs = distributions.gather(
            1,
            torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(-1),
        )
        loss_algo = self.compute_loss_algo(probs, advantages)
        # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 1)
        entropy_bonus = self.compute_entropy_bonus(distributions)
        total_loss = (
            loss_algo
            + self.coeff_loss_value * loss_value
            + self.entropy_coeff * entropy_bonus
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return loss_value, loss_algo

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

    gamma_ = 0.9

    optimizer_ = optim.RMSprop(
        list(value_network_.parameters()) + list(actor_network_.parameters()), lr=0.005
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
        test_every=100,
        epochs=1000,
        batch_size=128,
    )

    agent.train()
