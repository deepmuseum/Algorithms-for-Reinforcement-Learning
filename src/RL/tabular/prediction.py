import numpy as np
from RL.tabular.utils import bellman_operator


class Prediction:
    """
    Module to model the policy evaluation
    Attributes
    ----------
    env : object Environment
        the environment where the agent will operate
    policy : numpy array of dim n_states x n_actions
        policy[s][a] is the probability of taking action a given the agent is in a state s
    n_episodes: int
        number of episodes for learning the state values
    alpha: alpha
        learning rate
    """

    def __init__(self, env, policy, n_episodes, alpha):
        super().__init__()

        self.env = env
        self.policy = policy
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.V = np.zeros(env.Ns)  # initialize the state values

    def update(self, state, action, next_state, reward):
        raise NotImplementedError

    def run_online(self):
        """
        Perform policy evaluation in an online fashion
        """
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.policy[state]
                next_state, reward, done, info = self.env.step(action)
                self.update(state, action, next_state, reward)
                state = next_state

    def iterative_policy_evalution(self, epsilon=1e-3):
        """
        Perform policy evaluation by solving Bellman equation in an iterative way
        """
        raise NotImplementedError


class TD_prediction(Prediction):
    """
    Module to model the policy evaluation for TD(0)

    """

    def __init__(self, env, policy, n_episodes, alpha):
        super().__init__(env, policy, n_episodes, alpha)

    def update(self, state, action, next_state, reward):
        self.V[state] += self.alpha * (
            reward + self.env.gamma * self.V[next_state] - self.V[state]
        )
