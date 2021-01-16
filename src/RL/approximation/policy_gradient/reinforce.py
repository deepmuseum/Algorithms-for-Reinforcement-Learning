"""implementation of REINFORCE"""
import numpy as np
import torch
import torch.nn as nn
from RL.utils import make_seed

# TODO : use GPU & use tqdm


class PolicyModel(nn.Module):
    """
    Module to model the policy

    Attributes
    ----------
    n_actions : int
        number of possible actions
    dim_observation : int
        dimension of observations
    net : nn.Module
        neural network that models the policy
    """

    def __init__(self, dim_observation, n_actions, network):
        super().__init__()

        self.n_actions = n_actions
        self.dim_observation = dim_observation

        self.net = network

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        action = torch.multinomial(self.forward(state), 1)
        return action


class REINFORCE:
    """
    REINFORCE AGENT

    Attributes
    ----------
    env
        environment on which to learn the policy
    model : PolicyModel
        model of policy to learn
    gamma : float
        decay parameter
    optimizer
        optimization algorithm

    """

    def __init__(self, env, model, optimizer, seed, gamma):

        self.env = env
        make_seed(seed)
        self.env.seed(seed)
        self.model = model
        self.gamma = gamma

        self.optimizer = optimizer
        # self.monitor_env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)

    # Method to implement
    def _compute_returns(self, rewards):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            The array of rewards of one episode

        Returns
        -------
        array
            The cumulative discounted rewards at each time step

        Example
        -------
        for rewards=[1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3]
        """
        returns = []
        for t in range(len(rewards)):
            returns.append(
                np.sum(
                    [self.gamma ** (k - t) * rewards[k] for k in range(t, len(rewards))]
                )
            )
        return np.array(returns)

    def optimize_step(self, n_trajectories):
        """
        Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation

        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        reward_trajectories = []
        loss = torch.tensor(0, dtype=torch.float)
        for i in range(n_trajectories):
            rewards = []
            observation = self.env.reset()
            policy = []
            done = False
            while not done:
                observation = torch.tensor(observation, dtype=torch.float)
                action = self.model.select_action(observation)
                proba = self.model(observation)[int(action)]
                policy.append(proba)
                observation, reward, done, info = self.env.step(int(action))
                rewards.append(reward)
            returns = self._compute_returns(rewards)
            reward_trajectories.append(returns[0])

            for t in range(len(rewards)):
                # pseudo loss
                loss -= torch.log(policy[t]) * returns[t] / n_trajectories
        self.env.close()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.array(reward_trajectories)

    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates

        """

        rewards = []
        for episode in range(n_update):
            rewards.append(self.optimize_step(n_trajectories))
            print(
                f"Episode {episode + 1}/{n_update}: rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}"
            )

    def evaluate(self):
        """
        Evaluate the agent on a single trajectory
        """
        pass


class BaselineReinforce(REINFORCE):
    def __init__(
        self, env, model, optimizer_policy, seed, gamma, value_network, optimizer_value
    ):
        super().__init__(env, model, optimizer_policy, seed, gamma)
        self.value_network = value_network
        self.optimizer_value = optimizer_value
        self.criterion_value = nn.MSELoss()

    def optimize_step(self, n_trajectories):
        """
        Perform a gradient update using n_trajectories

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expectation

        Returns
        -------
        array
            The cumulative discounted rewards of each trajectory
        """
        reward_trajectories = []
        loss_policy, loss_value = (
            torch.tensor(0, dtype=torch.float),
            torch.tensor(0, dtype=torch.float),
        )
        for i in range(n_trajectories):
            rewards = []
            observation = self.env.reset()
            policy = []
            state_values = []
            done = False
            while not done:
                observation = torch.tensor(observation, dtype=torch.float)
                state_values.append(self.value_network(observation))
                action = self.model.select_action(observation)
                proba = self.model(observation)[int(action)]
                policy.append(proba)
                observation, reward, done, info = self.env.step(int(action))
                rewards.append(reward)
            returns = self._compute_returns(rewards)
            reward_trajectories.append(returns[0])

            for t in range(len(rewards)):
                # pseudo loss

                loss_policy -= self.gamma ** t * (
                    torch.log(policy[t])
                    * (returns[t] - state_values[t].item())
                    / n_trajectories
                )
                loss_value += self.gamma ** t * (
                    self.criterion_value(
                        torch.tensor([returns[t]], dtype=torch.float), state_values[t]
                    )
                    / n_trajectories
                )

        self.env.close()
        self.optimizer.zero_grad()
        self.optimizer_value.zero_grad()

        loss_policy.backward()
        loss_value.backward()
        self.optimizer.step()
        return np.array(reward_trajectories)


if __name__ == "__main__":

    from torch import optim
    import gym

    environment = gym.make("CartPole-v1")

    observations = environment.observation_space.shape[0]
    actions = environment.action_space.n
    net_policy = nn.Sequential(
        nn.Linear(in_features=observations, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=actions),
        nn.Softmax(dim=0),
    )

    net_value = nn.Sequential(
        nn.Linear(in_features=observations, out_features=16),
        nn.ReLU(),
        nn.Linear(in_features=16, out_features=8),
        nn.ReLU(),
        nn.Linear(in_features=8, out_features=1),
    )

    policy_model = PolicyModel(observations, actions, net_policy)

    opt_policy = optim.Adam(net_policy.parameters(), lr=0.01)
    opt_value = optim.Adam(net_value.parameters(), lr=0.01)

    agent = REINFORCE(
        env=environment, model=policy_model, gamma=1, seed=0, optimizer=opt_policy
    )

    # agent = BaselineReinforce(
    #     env=environment,
    #     model=policy_model,
    #     gamma=1,
    #     seed=0,
    #     optimizer_policy=opt_policy,
    #     optimizer_value=opt_value,
    #     value_network=net_value,
    # )

    agent.train(n_trajectories=50, n_update=50)
