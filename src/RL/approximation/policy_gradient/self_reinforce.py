"""implementation of REINFORCE"""
import numpy as np
import torch
import torch.nn as nn
import copy


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

    def select_action(self, state, epsilon=0.5):
        if np.random.rand() < epsilon:
            action = torch.randint(0, self.n_actions, (1,))
            return action
        action = torch.multinomial(self.forward(state), 1)
        return action

    def greedy_action(self, state):
        action = torch.argmax(self.forward(state))
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
        # make_seed(seed)
        # self.env.seed(seed)
        self.model = model
        self.gamma = gamma
        self.checkpoint = None
        self.optimizer = optimizer
        self.n_agents = len(self.env.reset())
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

    def optimize_step(self, n_trajectories, episode):
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
        reward_trajectories = [[] for _ in range(self.n_agents)]
        loss = torch.tensor(0, dtype=torch.float)

        for i in range(n_trajectories):
            rewards = [[] for _ in range(self.n_agents)]
            returns = [[] for _ in range(self.n_agents)]
            observations = self.env.reset()
            policy = [[] for _ in range(self.n_agents)]
            dones = [False] * self.n_agents

            while not np.any(dones):
                observations = [torch.tensor(observation, dtype=torch.float) for observation in observations]
                actions = [self.model.select_action(observation, epsilon=max(0.9 - episode * 0.001, 0.0)) for observation in observations]
                probas = [self.model(observation)[int(action)] for observation, action in zip(observations, actions)]
                observations_, rewards_, dones_ = self.env.step(actions)
                for k, (proba, reward,done) in enumerate(zip(probas, rewards_,dones_)):
                    if not dones_[k]:
                        rewards[k].append(reward)
                        policy[k].append(proba)
                    elif not dones[k]:
                        rewards[k].append(reward)
                        policy[k].append(proba)
                        dones[k]=True

            for k in range(self.n_agents):
                returns[k]+=list(self._compute_returns(rewards[k]))
                reward_trajectories[k].append(returns[k][0])

            for k in range(self.n_agents):
                for t in range(len(rewards[k])):
                    # pseudo loss
                    loss -= torch.log(policy[k][t]) * returns[k][t] / (n_trajectories * self.n_agents)
        self.env.close()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return [np.array(R) for R in reward_trajectories]

    def train(self, n_trajectories, n_update):
        """Training method

        Parameters
        ----------
        n_trajectories : int
            The number of trajectories used to approximate the expected gradient
        n_update : int
            The number of gradient updates

        """

        rewards = [[] for _ in range(self.n_agents)]
        rmax = 0
        for episode in range(n_update):
            reward_trajectories = self.optimize_step(n_trajectories, episode)
            print(
                f"Episode {episode + 1}/{n_update}:"
            )
            for k in range(self.n_agents):
                rewards[k].append(reward_trajectories[k])
                print("Agent "+str(k),f"rewards {round(rewards[k][-1].mean(), 2)} +/- {round(rewards[k][-1].std(), 2)}")
                if rewards[k][-1].mean() > rmax:
                    self.checkpoint = copy.deepcopy(self.model)
                    rmax = rewards[k][-1].mean()
                    print(rmax)
        print(rmax)

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
        self.checkpoint = None

    def optimize_step(self, n_trajectories, episode):
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
        reward_trajectories = [[] for _ in range(self.n_agents)]
        loss_policy, loss_value = (
            torch.tensor(0, dtype=torch.float),
            torch.tensor(0, dtype=torch.float),
        )
        for i in range(n_trajectories):
            rewards = [[] for _ in range(self.n_agents)]
            returns = [[] for _ in range(self.n_agents)]
            observations = self.env.reset()
            policy = [[] for _ in range(self.n_agents)]
            dones = [False] * self.n_agents
            state_values = [[] for _ in range(self.n_agents)]
            while not np.any(dones):
                observations = [torch.tensor(observation, dtype=torch.float) for observation in observations]
                actions = [self.model.select_action(observation, epsilon=max(0.9 - episode * 0.001, 0.0)) for
                           observation in observations]
                probas = [self.model(observation)[int(action)] for observation, action in zip(observations, actions)]
                observations_, rewards_, dones_ = self.env.step(actions)
                for k, (observation,proba, reward, done) in enumerate(zip(observations,probas, rewards_, dones_)):
                    if not dones_[k]:
                        state_values[k].append(self.value_network(observation))
                        rewards[k].append(reward)
                        policy[k].append(proba)
                    elif not dones[k]:
                        state_values[k].append(self.value_network(observation))
                        rewards[k].append(reward)
                        policy[k].append(proba)
                        dones[k] = True

            for k in range(self.n_agents):
                returns[k] += list(self._compute_returns(rewards[k]))
                reward_trajectories[k].append(returns[k][0])

            for k in range(self.n_agents):
                for t in range(len(rewards[k])):
                    # pseudo loss
                    loss_policy -= torch.log(policy[k][t]) * (returns[k][t]-- state_values[k][t].item()) / (n_trajectories * self.n_agents)
                    loss_value += self.gamma ** t * (
                        self.criterion_value(
                            torch.tensor([returns[k][t]], dtype=torch.float), state_values[k][t]
                        )
                        / (n_trajectories * self.n_agents)
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
