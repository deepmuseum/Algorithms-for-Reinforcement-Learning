import torch
import numpy as np
from torch import nn
from copy import deepcopy as c
from tqdm import tqdm
from RL.utils import PseudoEnv


class QNetwork(nn.Module):
    def __init__(self, model, n_a):
        super().__init__()
        self.model = model
        self.n_a = n_a

    def forward(self, state):
        return self.model(state)

    def compute_targets(self, rewards, gamma, states, optimal_actions, dones):
        values = self.forward(states)
        return rewards + (1 - dones) * gamma * values.gather(
            1, optimal_actions.unsqueeze(-1)
        ).squeeze(-1)

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
    ):

        self.q_network = q_network
        self.q_network.train()
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

    def choose_optimal_actions(self, states):
        values = self.copy_network(states)
        return values.argmax(dim=1)

    def build_targets_values(self, examples):
        states, actions, rewards, next_observations, dones = zip(*examples)
        # states = torch.tensor(states, dtype=torch.float, device=self.device)
        states = torch.cat(states, dim=0)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)
        values = self.q_network(states)  # shape batch_size, num_actions
        values = values.gather(
            -1, torch.tensor(actions, device=self.device).unsqueeze(-1)
        )
        values = values.squeeze(-1)
        # next_observations = torch.tensor(next_observations, dtype=torch.float, device=self.device)
        next_observations = torch.cat(next_observations, dim=0)
        optimal_next_actions = self.choose_optimal_actions(next_observations)
        targets = self.copy_network.compute_targets(
            rewards, self.gamma, next_observations, optimal_next_actions, dones
        )
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
                # observation_tensor = torch.tensor(observation, dtype=torch.float, device=self.device)
                action = self.q_network.eps_greedy(observation, self.epsilon)
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
                self.q_network.train()
                tk.set_postfix(
                    {
                        f"mean return": np.mean(returns),
                        "standard deviation": np.std(returns),
                    }
                )
            self.env.close()

    def evaluate(self):
        returns = []
        self.q_network.eval()
        with torch.no_grad():
            for _ in range(10):
                observation = self.env.reset()
                done = False
                episode_return = 0
                while not done:
                    # observation = torch.tensor(observation, dtype=torch.float, device=self.device)
                    action = self.q_network.greedy(observation)
                    observation, reward, done, _ = self.env.step(action)
                    episode_return += reward
                returns.append(episode_return)
                self.env.close()
            return returns


class DDQN(DQN):
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
        steps_target=5,
    ):
        super().__init__(
            env,
            q_network,
            gamma,
            batch_size,
            optimizer,
            device,
            num_episodes,
            max_size_buffer,
            steps_target,
        )

    def choose_optimal_actions(self, states):
        values = self.q_network(states)
        return values.argmax(dim=1)


if __name__ == "__main__":
    import gym
    from torch import optim
    import torch.nn.functional as F

    device = torch.device("cpu")

    environment_ = gym.make(
        "CartPole-v0"
    ).unwrapped  # 'BreakoutDeterministic-v4' "Pong-v0" "CartPole-v1"
    environment = PseudoEnv(environment_, device)
    environment.reset()
    dim_observations = (environment.observation_dim[0], environment.observation_dim[1])
    num_actions = environment.env.action_space.n
    print(f"observation dimension {dim_observations}")
    print(f"number of actions {num_actions}")

    # q_model = nn.Sequential(
    #     nn.Linear(in_features=observations, out_features=16),
    #     nn.ReLU(),
    #     nn.Linear(in_features=16, out_features=8),
    #     nn.ReLU(),
    #     nn.Linear(in_features=8, out_features=num_actions),
    # )

    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 6, 5)
            # we use the maxpool multiple times, but define it once
            self.pool = nn.MaxPool2d(2, 2)
            # in_channels = 6 because self.conv1 output 6 channel
            self.conv2 = nn.Conv2d(6, 10, 5)
            # 5*5 comes from the dimension of the last convnet layer
            self.conv3 = nn.Conv2d(10, 16, 5)
            self.fc1 = nn.Linear(576, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_actions)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 576)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)  # no activation on final layer
            return x

    class DQNModel(nn.Module):
        def __init__(self, h, w, outputs):
            super(DQNModel, self).__init__()
            self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)

            # Number of Linear input connections depends on output of conv2d layers
            # and therefore the input image size, so compute it.
            def conv2d_size_out(size, kernel_size=5, stride=2):
                return (size - (kernel_size - 1) - 1) // stride + 1

            convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
            convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
            linear_input_size = convw * convh * 32
            self.head = nn.Linear(linear_input_size, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return self.head(x.view(x.size(0), -1))

    print(dim_observations)
    print(num_actions)
    q_model = DQNModel(*dim_observations, num_actions)

    opt = optim.Adam(q_model.parameters())

    q_net = QNetwork(model=q_model, n_a=num_actions).to(device)
    g = 0.999
    bsz = 100
    num_ep = 1000
    max_size = bsz * 100
    agent = DQN(
        env=environment,
        q_network=q_net,
        gamma=g,
        batch_size=bsz,
        optimizer=opt,
        device=device,
        num_episodes=num_ep,
        max_size_buffer=max_size,
    )

    agent.train()
