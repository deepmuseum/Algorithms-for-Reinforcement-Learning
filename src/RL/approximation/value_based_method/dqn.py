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

    def compute_targets(self, rewards, gamma, states, dones):
        values = self.forward(states)
        return rewards + (1 - dones) * gamma * values.max(dim=1).values

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
        steps_target=20,
    ):

        self.q_network = q_network
        self.copy_network = c(q_network)
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.memory_buffer = []
        self.max_size_buffer = max_size_buffer
        # TODO : use GPU
        self.device = device
        self.num_episodes = num_episodes
        self.criterion = nn.MSELoss()
        self.epsilon = 0.1
        self.step_target = steps_target

    def sample_from_buffer(self):
        if len(self.memory_buffer) >= self.batch_size:
            indices = np.random.choice(
                range(len(self.memory_buffer)), size=self.batch_size, replace=False
            )
            return [self.memory_buffer[i] for i in indices]
        else:
            return self.memory_buffer

    def build_targets_values(self, examples):
        states, actions, rewards, next_observations, dones = zip(*examples)
        states = torch.tensor(states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        values = self.q_network(states)  # shape batch_size, num_actions
        values = values.gather(-1, torch.tensor(actions).unsqueeze(-1))
        values = values.squeeze(-1)
        targets = self.copy_network.compute_targets(rewards, self.gamma, states, dones)
        return targets, values

    def train(self):
        tk = tqdm(range(self.num_episodes), unit="episode")
        for episode in tk:
            # if (episode + 1) % (self.num_episodes // 10) == 0:
            #     self.epsilon /= 2
            observation = self.env.reset()
            done = False
            total_return = 0
            steps = 0
            while not done:
                steps += 1
                observation_tensor = torch.tensor(observation, dtype=torch.float)
                action = self.q_network.eps_greedy(observation_tensor, self.epsilon)
                next_observation, reward, done, _ = self.env.step(action)
                total_return += reward
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
                self.optimizer.step()
                if steps == self.step_target:
                    self.copy_network.load_state_dict(self.q_network.state_dict())
                    steps = 0
                observation = next_observation

            if (episode + 1) % 10 == 0:
                returns = self.evaluate()
                tk.set_postfix(
                    {
                        f"mean return": np.mean(returns),
                        "standard deviation": np.std(returns),
                    }
                )
            self.env.close()

    def evaluate(self):
        returns = []
        for i in range(10):
            observation = self.env.reset()
            done = False
            episode_return = 0
            while not done:
                observation = torch.tensor(observation, dtype=torch.float)
                action = self.q_network.greedy(observation)
                observation, reward, done, _ = self.env.step(action)
                episode_return += reward
            returns.append(episode_return)
            self.env.close()
        return returns


if __name__ == "__main__":
    import gym
    from torch import optim

    environment = gym.make("CartPole-v1")
    observations = environment.observation_space.shape[0]
    num_actions = environment.action_space.n

    q_model = nn.Sequential(
        nn.Linear(in_features=observations, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=num_actions),
    )

    opt = optim.Adam(q_model.parameters(), lr=0.001)

    q_net = QNetwork(model=q_model, n_a=num_actions)
    g = 1
    bsz = 100
    num_ep = 2000
    max_size = bsz * 10
    agent = DQN(
        env=environment,
        q_network=q_net,
        gamma=g,
        batch_size=bsz,
        optimizer=opt,
        device=torch.device("cpu"),
        num_episodes=num_ep,
        max_size_buffer=max_size,
    )

    agent.train()
