"""implementation of REINFORCE"""
import numpy as np
import torch
import torch.nn as nn
from RL.utils import make_seed
from tqdm import tqdm 
from copy import deepcopy as c
import pickle 


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
        return self.net(state).squeeze(0)



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

    def __init__(self, env, model, optimizer, gamma, device, path, test_every):
        self.env = env
        self.device = device
        self.model = model.to(self.device)
        self.gamma = gamma
        self.epsilon = -1
        self.optimizer = optimizer
        self.best_return = - float("inf") 
        self.best_state_dict = c(self.model.cpu().state_dict())
        self.model.to(self.device)
        self.path = path
        self.test_every = test_every

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
        rewards = [1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3]
        """
        returns = []
        for t in range(len(rewards)):
            returns.append(
                np.sum(
                    [self.gamma ** (k - t) * rewards[k] for k in range(t, len(rewards))]
                )
            )
        return np.array(returns)

    def select_from_prob(self, dist):
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
        if np.random.uniform() >= self.epsilon: 
            action = int(torch.multinomial(dist, 1))
        else:
            action = np.random.choice(range(self.model.n_actions))
        return action

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
        return_trajectories = []
        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        for i in range(n_trajectories):
            rewards = []
            observation = self.env.reset()
            policy = []
            done = False
            while not done:
                observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
                prob_dist = self.model(observation).squeeze(0)
                action = self.select_from_prob(prob_dist)
                proba = prob_dist[action]
                policy.append(proba)
                observation, reward, done, info = self.env.step(action)
                rewards.append(reward)
            returns = self._compute_returns(rewards)
            return_trajectories.append(returns[0])

            for t in range(len(rewards)):
                # pseudo loss
                loss -= torch.log(policy[t]) * returns[t] / n_trajectories
        self.env.close()
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        return np.array(return_trajectories)

    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates

        """
        tk = tqdm(range(n_update), unit="update")
        for update in tk:
            self.epsilon = max(0.9 - update * 0.001, 0.0)
            self.optimize_step(n_trajectories)
            if (update + 1) % self.test_every == 0:
                  returns = self.evaluate()
                  tk.set_postfix({"mean reward": round(returns.mean(), 2), "std": round(returns.std(), 2)})
                  if returns.mean() >= self.best_return:
                      self.best_return = returns.mean()
                      self.best_state_dict = c(self.model.cpu().state_dict())
                      self.model.to(self.device)
                      pickle.dump(self.best_state_dict, open(self.path, "wb"))

    def evaluate(self):
        """
        Evaluate the agent on a 50 trajectory
        """
        returns = []
        for _ in range(50):
            done = False
            observation = self.env.reset()
            r = 0
            while not done:
                observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    prob_dist = self.model(observation).squeeze(0)
                action = int(torch.argmax(prob_dist))
                observation, reward, done, info = self.env.step(action)
                r += reward
            returns.append(r)
        return np.array(returns)


class REINFORCEWithBaseline(REINFORCE):
    def __init__(
        self, env, model, optimizer_policy, gamma, device, value_network, optimizer_value
    ):
        super().__init__(env, model, optimizer_policy, gamma, device)
        self.value_network = value_network.to(self.device)
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
            torch.tensor(0, dtype=torch.float, device=self.device),
            torch.tensor(0, dtype=torch.float, device=self.device),
        )
        for i in range(n_trajectories):
            rewards = []
            observation = self.env.reset()
            policy = []
            state_values = []
            done = False
            while not done:
                observation = torch.tensor(observation, dtype=torch.float, device=self.device).unsqueeze(0)
                state_values.append(self.value_network(observation).squeeze(0))
                prob_dist = self.model(observation).squeeze(0)
                action = self.select_from_prob(prob_dist)
                proba = prob_dist[action]
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
                        torch.tensor([returns[t]], dtype=torch.float, device=self.device), state_values[t]
                    )
                    / n_trajectories
                )

        self.env.close()
        self.optimizer.zero_grad()
        self.optimizer_value.zero_grad()

        loss_policy.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        loss_value.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.optimizer_value.step()
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
        nn.BatchNorm(16),
        nn.Linear(in_features=16, out_features=8),
        nn.ReLU(),
        nn.BatchNorm(8),
        nn.Linear(in_features=8, out_features=actions),
        nn.Softmax(dim=1),
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

    device = torch.device('cuda')
    # agent = REINFORCE(
    #     env=environment, model=policy_model, gamma=1, seed=0, optimizer=opt_policy, device=device
    # )

    agent = REINFORCEWithBaseline(
        env=environment,
        model=policy_model,
        gamma=1,
        optimizer_policy=opt_policy,
        optimizer_value=opt_value,
        value_network=net_value,
        device=device
    )

    agent.train(n_trajectories=100, n_update=100)
