import numpy as np


def bellman_operator(V, policy, R, P, gamma):
    """

    Parameters
    ----------
    V
    policy
    R
    P
    gamma

    Returns
    -------

    """
    raise NotImplementedError


def eps_greedy(state, n_actions, Q, epsilon):
    """

    Parameters
    ----------
    state
    n_actions
    Q
    epsilon

    Returns
    -------

    """
    if np.random.uniform() <= epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(Q[state])


class Control:
    def __init__(self, env, n_episodes=10, alpha=0.01):
        """
        Module to model the policy evaluation
        Attributes
        ----------
        env
            the environment where the agent will operate
        n_episodes: int
            number of episodes for learning the policy
        alpha: float
            learning rate
        """
        super().__init__()
        self.env = env
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.V = np.zeros(env.Ns)  # initialize the state values
        self.Q = np.zeros((env.Ns, env.Na))
        self.policy = np.zeros((self.env.Ns, self.env.Na))

    def behave(self, state):
        raise NotImplementedError

    def behavior_policy(self):
        raise NotImplementedError

    def target_policy(self):
        raise NotImplementedError

    def sample_action(self, state, policy):
        return np.random.choice(np.arange(self.env.Na), p=policy[state])

    def compute_returns(self, rewards):
        gammas = np.array([self.env.gamma ** i for i in range(len(rewards))])
        return np.sum(gammas * np.array(rewards))

    def update_online(self, state, action, next_state, reward):
        """
        Given the current state, current action, the reward and the next state,
        performs an online update in each step of an episode
        """
        raise NotImplementedError

    def update_offline(self, trajectory):
        """
        Given states, actions and rewards collected during a whole trajectory, performs an offline update
        """
        raise NotImplementedError

    def run_online(self):
        """
        Estimate optimal action state values in an online fashion
        then compute the greedy (deterministic) policy from the estimated optimal Q function
        """
        for episode in range(self.n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.behave(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_online(state, action, next_state, reward)
                state = next_state
        for state in range(self.env.Ns):
            self.policy[state][np.argmax(self.Q[state])] = 1

    def run_offline(self):
        for episode in range(self.n_episodes):
            trajectory = {"states": [], "actions": [], "rewards": []}
            done = False
            state = self.env.reset()
            while not done:
                action = self.sample_action(state, self.policy)
                next_state, reward, done, info = self.env.step(action)
                trajectory["actions"].append(action)
                trajectory["states"].append(state)
                trajectory["rewards"].append(reward)
                state = next_state
            self.update_offline(trajectory)
            self.policy *= 0
            for state in range(self.env.Ns):
                self.policy[state][np.argmax(self.Q[state])] = 1


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

    def iterative_policy_evaluation(self, epsilon=1e-3):
        """
        Perform policy evaluation by solving Bellman equation in an iterative way
        """
        raise NotImplementedError
