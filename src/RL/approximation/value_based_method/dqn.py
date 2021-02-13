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
        action_map=None,
    ):

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
                observation_tensor = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                )
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
                observation = torch.tensor(
                    observation, dtype=torch.float, device=self.device
                )
                action = self.q_network.greedy(observation)
                observation, reward, done, _ = self.env.step(action)
                episode_return += reward
            returns.append(episode_return)
            self.env.close()
        return returns


if __name__ == "__main__":

    from torch import optim
    from kaggle_environments import make
    from kaggle_environments.envs.hungry_geese.hungry_geese import (
        Observation,
        Configuration,
        Action,
        row_col,
    )

    # What's necessary ?
    # step function, that takes the action in entry and return observation reward and done
    # observation must be a list of floats
    # make a grid representing the state of the game :
    # 1 -> body
    # 2 -> head
    # 3 -> food
    # 0 -> nothing
    # 4 -> ennemy
    # the grid is first flattened, then maybe we'll use CNNs

    # close
    # reset

    def enemy_toward(obs_dict, config_dict):
        """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
        observation = Observation(obs_dict)
        configuration = Configuration(config_dict)
        player_index = observation.index
        player_goose = observation.geese[player_index]
        player_head = player_goose[0]
        player_row, player_column = row_col(player_head, configuration.columns)
        food = observation.food[0]
        food_row, food_column = row_col(food, configuration.columns)

        if food_row > player_row:
            return Action.SOUTH.name
        if food_row < player_row:
            return Action.NORTH.name
        if food_column > player_column:
            return Action.EAST.name
        return Action.WEST.name

    class PseudoEnvGeese:
        def __init__(self, enemy=enemy_toward):
            self.env = make("hungry_geese", debug=False)
            self.trainer = self.env.train([None, enemy])
            self.grid = np.zeros(
                (self.env.configuration["rows"], self.env.configuration["columns"])
            )
            self.configuration = Configuration(self.env.configuration)
            self.action_map = {i: action.name for i, action in enumerate(Action)}

        def step(self, action):
            self.grid = np.zeros(
                (self.env.configuration["rows"], self.env.configuration["columns"])
            )
            obs, reward, done, _ = self.trainer.step(self.action_map[action])
            obs = Observation(obs)
            player_index = obs.index
            player_goose = obs.geese[player_index]
            if len(player_goose) > 0:
                player_head = player_goose[0]
                player_head_row, player_head_column = row_col(
                    player_head, self.configuration.columns
                )
                self.grid[player_head_row, player_head_column] = 2
                if len(player_goose) > 1:
                    player_body = player_goose[1:]
                    rows, cols = [], []
                    for elt in player_body:
                        row, col = row_col(elt, self.configuration.columns)
                        rows.append(row)
                        cols.append(col)
                    self.grid[rows, cols] = 1

            foods = obs.food
            for food in foods:
                food_row, food_column = row_col(food, self.configuration.columns)
                self.grid[food_row, food_column] = 3
            enemies = [obs.geese[i] for i in range(len(obs.geese)) if i != player_index]
            if len(enemies) > 0:
                rows, cols = [], []
                for enemy in enemies:
                    for elt in enemy:
                        row, col = row_col(elt, self.configuration.columns)
                        rows.append(row)
                        cols.append(col)
                self.grid[rows, cols] = -1

            return self.grid.flatten().tolist(), reward, done, _

        def reset(self):
            obs = self.trainer.reset()
            self.grid = np.zeros(
                (self.env.configuration["rows"], self.env.configuration["columns"])
            )
            obs = Observation(obs)
            player_index = obs.index
            player_goose = obs.geese[player_index]
            if len(player_goose) > 0:
                player_head = player_goose[0]
                player_head_row, player_head_column = row_col(
                    player_head, self.configuration.columns
                )
                self.grid[player_head_row, player_head_column] = 2
                if len(player_goose) > 1:
                    player_body = player_goose[1:]
                    rows, cols = [], []
                    for elt in player_body:
                        row, col = row_col(elt, self.configuration.columns)
                        rows.append(row)
                        cols.append(col)
                    self.grid[rows, cols] = 1

            foods = obs.food
            for food in foods:
                food_row, food_column = row_col(food, self.configuration.columns)
                self.grid[food_row, food_column] = 3
            enemies = [obs.geese[i] for i in range(len(obs.geese)) if i != player_index]
            if len(enemies) > 0:
                rows, cols = [], []
                for enemy in enemies:
                    for elt in enemy:
                        row, col = row_col(elt, self.configuration.columns)
                        rows.append(row)
                        cols.append(col)
                self.grid[rows, cols] = -1

            return self.grid.flatten().tolist()

        def close(self):
            pass

    environment = PseudoEnvGeese()
    observations = environment.configuration.columns * environment.configuration.rows
    num_actions = 4
    q_model = nn.Sequential(
        nn.Linear(in_features=observations, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=num_actions),
    )

    opt = optim.Adam(q_model.parameters(), lr=0.01)

    q_net = QNetwork(model=q_model, n_a=num_actions)
    g = 1
    bsz = 100
    num_ep = 1000
    max_size = bsz * 10

    device = torch.device("cuda:0")
    q_net = q_net.to(device)
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
