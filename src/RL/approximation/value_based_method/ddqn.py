from RL.approximation.value_based_method.dqn import DQN


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
        steps_target=1000,
        evaluate_every=5000,
        exploration=0.1,
        path=None,
        num_agents=1,
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
            evaluate_every=evaluate_every,
            exploration=exploration,
            path=path,
            num_agents=num_agents,
        )

    def choose_optimal_actions(self, states):
        values = self.q_network(states)
        return values.argmax(dim=1)


if __name__ == "__main__":
    import torch
    import gym
    from torch import optim, nn
    from RL.approximation.value_based_method.dqn import QNetwork

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

    opt = optim.Adam(q_model.parameters(), lr=0.01)

    q_net = QNetwork(model=q_model, n_a=num_actions)
    g = 1
    bsz = 100
    num_ep = 1000
    max_size = bsz * 10
    agent = DDQN(
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
