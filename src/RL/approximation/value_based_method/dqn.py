import torch
import numpy as np
from torch import nn
from copy import deepcopy as c
from tqdm import tqdm
import pickle
import inspect


class QNetwork(nn.Module):
    def __init__(self, model, n_a):
        super().__init__()
        self.model = model
        self.n_a = n_a

    def forward(self, state):
        return self.model(state)

    def eps_greedy(self, state, epsilon, num_agents):
        if np.random.uniform() <= epsilon:
            return np.random.randint(0, self.n_a, size=(num_agents,))
        else:
            values = self.forward(state).squeeze(0)
            return torch.argmax(values, dim=-1).tolist()

    def greedy(self, state):
        values = self.forward(state).squeeze(0)
        return torch.argmax(values, dim=-1).tolist()


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
        path=None,
        num_agents=1,
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
        self.path = path
        self.num_agents = num_agents

    def sample_from_buffer(self):
        if len(self.memory_buffer) >= self.batch_size:
            indices = np.random.choice(
                range(len(self.memory_buffer)), size=self.batch_size, replace=False
            )
            return [self.memory_buffer[i] for i in indices]
        else:
            return self.memory_buffer

    def compute_targets(self, rewards, states, dones, dones_prev):
        values = self.copy_network(states)  # shape bsz, num_agents, num_actions
        optimal_actions = values.argmax(dim=-1)  # shape bsz, num_agents
        return (1 - dones_prev) * rewards + (1 - dones) * self.gamma * values.gather(
            -1, optimal_actions.unsqueeze(1)  # unsqueeze 1 or -1 ?
        ).squeeze(-1)

    def build_targets_values(self, examples):
        states, actions, rewards, next_observations, dones, dones_prev = zip(*examples)
        states = torch.tensor(
            states, dtype=torch.float, device=self.device
        )  # shape (bsz, num_agents, dim_obs)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device
        )  # shape (bsz, num_agents)
        dones_prev = torch.tensor(dones_prev, dtype=torch.float, device=self.device)
        dones = torch.tensor(
            dones, dtype=torch.float, device=self.device
        )  # shape (bsz, num_agents)
        values = self.q_network(states)  # shape batch_size, num_agents, num_actions,
        values = values.gather(
            -1,
            torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(
                1
            ),  # shape batch_size, num_agents
        )  # shape batch_size, num_agents, 1
        values = (1 - dones_prev) * values.squeeze(
            -1
        )  # shape batch_size, num_agents (avoid impacting the loss by dead agents)
        next_observations = torch.tensor(
            next_observations, dtype=torch.float, device=self.device
        )  # shape (bsz, num_agents, dim_obs)
        targets = self.compute_targets(
            rewards, next_observations, dones, dones_prev
        )  # shape (bsz, num_agents)
        return targets.view(targets.shape[0], -1), values.view(values.shape[0], -1)

    def train(self):
        tk = tqdm(range(self.num_episodes), unit="episodes")
        for episode in tk:

            observation = (
                self.env.reset()
            )  # observation is a list of of lists of len num_agents
            if self.num_agents == 1:
                observation = [observation]
            done_prev = [False for _ in range(self.num_agents)]
            done = [False for _ in range(self.num_agents)]
            steps = 0
            while not all(done):
                steps += 1
                observation_tensor = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                ).unsqueeze(
                    0
                )  # shape 1, num_agents, dim_obs
                action = self.q_network.eps_greedy(
                    observation_tensor, self.epsilon, self.num_agents
                )
                if self.num_agents == 1 and hasattr(action, "__len__"):
                    action = action[0]
                # list of int of len num_agents
                next_observation, reward, done, _ = self.env.step(action)
                if self.num_agents == 1:
                    next_observation, reward, done, action = (
                        [next_observation],
                        [reward],
                        [done],
                        [action],
                    )
                # 3 lists of len num_agents
                self.memory_buffer.append(
                    (observation, action, reward, next_observation, done, done_prev)
                )
                done_prev = done
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
                mean = np.mean(returns, axis=0)
                std = np.std(returns, axis=0)
                tk.set_postfix(
                    {
                        f"agent {i} (mean, std)": (mean[i], std[i])
                        for i in range(self.num_agents)
                    }
                )
                if np.mean(mean) >= self.best_return:
                    self.best_return = np.mean(mean)
                    if self.path is not None:
                        self.best_state_dict = c(self.q_network.cpu().state_dict())
                        self.q_network.to(self.device)
                        pickle.dump(self.best_state_dict, open(self.path, "wb"))

            self.env.close()

    def evaluate(self):
        returns = []
        has_val_param = "val" in inspect.signature(self.env.step).parameters

        for _ in range(100):
            observation = self.env.reset()
            if self.num_agents == 1:
                observation = [observation]
            done = [False for _ in range(self.num_agents)]
            episode_return = [0 for _ in range(self.num_agents)]
            while not all(done):
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                )
                action = self.q_network.greedy(observation.unsqueeze(0))
                if self.num_agents == 1:
                    action = action[0]
                if has_val_param:
                    observation, reward, done, _ = self.env.step(action, val=True)
                else:
                    observation, reward, done, _ = self.env.step(action)
                if self.num_agents == 1:
                    observation, reward, done = [observation], [reward], [done]
                for j, r in enumerate(reward):
                    episode_return[j] += r
            returns.append(episode_return)  # shape 100, num_agents
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

    device_ = torch.device("cuda")
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
