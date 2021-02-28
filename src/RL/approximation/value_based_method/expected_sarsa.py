import torch
from torch import nn
import torch.nn.functional as F
from RL.approximation.value_based_method.dqn import DQN, QNetwork


class ExpectedSarsa(DQN):
    def __init__(
        self,
        env,
        q_network,
        optimizer,
        device,
        num_episodes,
        gamma,
        max_size_buffer,
        batch_size,
        steps_target,
        evaluate_every,
        exploration,
    ):

        super().__init__(
            env=env,
            q_network=q_network,
            optimizer=optimizer,
            device=device,
            gamma=gamma,
            num_episodes=num_episodes,
            max_size_buffer=max_size_buffer,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            exploration=exploration,
            steps_target=steps_target,
        )

    def compute_targets(self, rewards, states, dones):
        values = self.copy_network(states)
        optimal_actions = values.argmax(dim=-1)
        eps_greedy_policy = (1 - self.epsilon) * F.one_hot(
            optimal_actions, num_classes=values.shape[-1]
        ) + self.epsilon / values.shape[-1] * torch.ones(
            values.shape, device=self.device
        )
        expectations = torch.sum(values * eps_greedy_policy, dim=-1, keepdim=False)
        return rewards + (1 - dones) * self.gamma * expectations


if __name__ == "__main__":
    from torch import optim
    import gym

    environment = gym.make("CartPole-v1")

    value_net = nn.Sequential(
        nn.Linear(environment.observation_space.shape[0], 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, environment.action_space.n),
    )

    q_net = QNetwork(value_net, environment.action_space.n)

    gamma_ = 0.99
    optimizer_ = optim.Adam(q_net.parameters(), lr=0.001)

    device_ = torch.device("cpu")
    bsz = 100
    num_ep = 1000
    max_size = bsz * 10
    steps_target_ = 10
    evaluate_every_ = 100
    exploration_ = 0.1

    agent = ExpectedSarsa(
        env=environment,
        q_network=q_net,
        optimizer=optimizer_,
        device=device_,
        gamma=gamma_,
        num_episodes=num_ep,
        max_size_buffer=max_size,
        batch_size=bsz,
        evaluate_every=evaluate_every_,
        exploration=exploration_,
        steps_target=steps_target_,
    )

    agent.train()
