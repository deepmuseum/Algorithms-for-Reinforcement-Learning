import torch
import numpy as np
from torch import nn
from copy import deepcopy as c
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self, model, n_a):
        super().__init__()
        self.model = model
        self.n_a = n_a

    def forward(self, state):
        return self.model(state)

    def eps_greedy(self, state, epsilon):
        if np.random.uniform() <= epsilon:
            return np.random.randint(self.n_a)
        else:
            values = self.forward(state)
            return torch.argmax(values).item()

    def greedy(self, state):
        values = self.forward(state)
        return torch.argmax(values).item()


class DQN:
    """
    Implementation of Deep Q-learning

    """

    def __init__(
        self,
        env,
        q_network,
        gamma,
        batch_size,
        optimizer,
        device,
        num_episodes,
        max_size_buffer,
        steps_target=10,
        evaluate_every=1000,
        exploration=0.1,
    ):
        self.exploration = exploration
        self.evaluate_every = evaluate_every
        self.q_network = q_network
        self.copy_network = c(q_network)
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.memory_buffer = []
        self.max_size_buffer = max_size_buffer
        self.device = device
        self.num_episodes = num_episodes
        self.criterion = nn.MSELoss()
        self.epsilon = 1
        self.step_target = steps_target
        self.best_return = 0
        self.best_state_dict = c(self.q_network.state_dict())

    def sample_from_buffer(self):
        if len(self.memory_buffer) >= self.batch_size:
            indices = np.random.choice(
                range(len(self.memory_buffer)), size=self.batch_size, replace=False
            )
            return [self.memory_buffer[i] for i in indices]
        else:
            return self.memory_buffer

    def compute_targets(self, rewards, states, dones):
        values = self.copy_network(states)
        optimal_actions = values.argmax(dim=-1)
        return rewards + (1 - dones) * self.gamma * values.gather(
            1, optimal_actions.unsqueeze(-1)
        ).squeeze(-1)

    def build_targets_values(self, examples):
        states, actions, rewards, next_observations, dones = zip(*examples)
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
        targets = self.compute_targets(rewards, next_observations, dones)
        return targets, values

    def train(self):
        tk = tqdm(range(self.num_episodes), unit="episode")
        for episode in tk:

            observation = self.env.reset()
            done = False
            steps = 0
            while not done:
                steps += 1
                observation_tensor = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).unsqueeze(0)
                action = self.q_network.eps_greedy(observation_tensor, self.epsilon)
                next_observation, reward, done, _ = self.env.step(action)
                self.memory_buffer.append(
                    (observation, action, reward, next_observation, done)
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

    def evaluate(self):
        returns = []
        for i in range(100):
            observation = self.env.reset()
            done = False
            episode_return = 0
            while not done:
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                )
                action = self.q_network.greedy(observation.unsqueeze(0))
                observation, reward, done, _ = self.env.step(action)
                episode_return += reward
            returns.append(episode_return)
            self.env.close()
        return returns


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

    agent = DQN(
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
