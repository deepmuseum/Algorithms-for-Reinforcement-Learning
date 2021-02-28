import torch
from torch import nn
from RL.approximation.value_based_method.dqn import DQN, QNetwork
from tqdm import tqdm
import numpy as np
from copy import deepcopy as c


class Sarsa(DQN):
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

    def train(self):
        tk = tqdm(range(self.num_episodes), unit="episode")
        for episode in tk:
            observation = self.env.reset()
            observation_tensor = torch.tensor(
                observation, dtype=torch.float, device=self.device
            ).unsqueeze(0)
            action = self.q_network.eps_greedy(observation_tensor, self.epsilon)
            next_observation, reward, done, _ = self.env.step(action)
            steps = 0

            while not done:
                steps += 1
                next_observation_tensor = torch.tensor(
                    next_observation, dtype=torch.float, device=self.device
                ).unsqueeze(0)
                next_action = self.q_network.eps_greedy(
                    next_observation_tensor, self.epsilon
                )
                self.memory_buffer.append(
                    (observation, action, reward, next_observation, next_action, done)
                )
                if len(self.memory_buffer) > self.max_size_buffer:
                    self.memory_buffer.pop(0)
                examples = self.sample_from_buffer()
                targets, values = self.build_targets_values(examples)
                loss = self.criterion(targets, values)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
                self.optimizer.step()
                if steps == self.step_target:
                    self.copy_network.load_state_dict(self.q_network.state_dict())
                    steps = 0
                observation = next_observation
                action = next_action
                next_observation, reward, done, _ = self.env.step(action)

            # decrease linearly epsilon the first x % episodes from 1 until .1
            if (episode + 1) <= int(self.num_episodes * self.exploration):
                self.epsilon -= 0.9 / int(self.num_episodes * self.exploration)
            else:
                self.epsilon = 0.1  # just to be sure
            if (episode + 1) % self.evaluate_every == 0:
                returns = self.evaluate()
                mean = np.mean(returns)
                tk.set_postfix(
                    {f"mean return": mean, "standard deviation": np.std(returns)}
                )
                if mean >= self.best_return:
                    self.best_return = mean
                    self.best_state_dict = c(self.q_network.state_dict())

            self.env.close()

    def build_targets_values(self, examples):
        states, actions, rewards, next_observations, next_actions, dones = zip(
            *examples
        )
        states = torch.tensor(states, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        values = self.q_network(states)  # shape batch_size, num_actions
        values = values.gather(
            -1, torch.tensor(actions, device=self.device).unsqueeze(-1)
        )
        values = values.squeeze(-1)
        next_observations = torch.tensor(
            next_observations, dtype=torch.float, device=self.device
        )
        next_actions = torch.tensor(actions, device=self.device).unsqueeze(-1)
        targets = self.compute_targets_sarsa(
            rewards, next_observations, next_actions, dones
        )
        return targets, values

    def compute_targets_sarsa(self, rewards, states, actions, dones):
        values = self.copy_network(states)
        return rewards + (1 - dones) * self.gamma * values.gather(1, actions).squeeze(
            -1
        )


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
    bsz = 1000
    num_ep = 10000
    max_size = bsz * 10
    steps_target_ = 500
    evaluate_every_ = 100
    exploration_ = 0.1

    agent = Sarsa(
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
