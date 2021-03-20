"""
https://arxiv.org/pdf/1707.06347.pdf
"""
from RL.approximation.policy_gradient.a2c import A2CAgent
from copy import deepcopy as c
import torch
import numpy as np


class PPO(A2CAgent):
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
        epsilon=0.2,
    ):
        super().__init__(
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
            entropy_coeff,
            coeff_loss_value,
            num_agents,
        )

        self.old_actor = c(self.actor_network)
        self.epsilon = epsilon

    def compute_loss_ppo(self, probs, advantages, old_probs):
        ratios = torch.exp(torch.log(probs) - torch.log(old_probs).detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def optimize_step(self, observations, actions, targets, advantages):
        self.old_actor.load_state_dict(c(self.actor_network.state_dict()))
        index = np.random.permutation(
            range(self.timesteps)
        )
        for _ in range(self.epochs):
            for j in range(0, self.timesteps, self.batch_size):
                values = self.value_network(
                    observations[index[j : j + self.batch_size]]
                ).squeeze(
                    -1
                )  # shape (bsz, num_agents)

                # Compute returns and advantages

                # Learning step !
                self.optimize_batch(
                    values,
                    targets[index[j : j + self.batch_size]],
                    advantages[index[j : j + self.batch_size]],
                    torch.tensor(actions, device=self.device, dtype=torch.int64)[
                        index[j : j + self.batch_size]
                    ],
                    observations[index[j : j + self.batch_size]],
                )

    def optimize_batch(self, values, targets, advantages, actions, observations):
        self.optimizer.zero_grad()
        loss_value = self.compute_loss_value(values, targets)
        distributions = self.actor_network(
            observations
        )  # shape (bsz, num_agents, num_actions)

        probs = distributions.gather(
            -1,
            actions.unsqueeze(-1),
            # actions is shape (bsz, num_agents) so unsqueeze -1
        ).squeeze(
            -1
        )  # probs is shape  (bsz, num_agents)

        # probs from previous policy
        with torch.no_grad():
            prev_dist = self.old_actor(observations)
            prev_probs = prev_dist.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        loss_algo = self.compute_loss_ppo(
            probs, advantages, prev_probs
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


if __name__ == "__main__":
    from RL.approximation.policy_gradient.a2c import ActorNetwork, ValueNetwork
    import torch.nn as nn
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
    agent = PPO(
        env=environment,
        actor_network=actor_network_,
        value_network=value_network_,
        gamma=gamma_,
        optimizer=optimizer_,
        device=device_,
        n_a=environment.action_space.n,
        path=None,
        test_every=10,
        epochs=4,
        batch_size=128,
        timesteps=1000,
        updates=1000,
    )

    agent.train()
