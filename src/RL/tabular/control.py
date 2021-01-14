import numpy as np
from RL.tabular.utils import eps_greedy


class Control:
    def __init__(self, env, n_episodes=10, alpha=0.01):
        """
        Module to model the policy evaluation
        Attributes
        ----------
        env : object Environment
            the environment where the agent will operate
        n_episodes: int
            number of episodes for learning the policy
        alpha: alpha
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

    def update(self, state, action, next_state, reward):
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
                self.update(state, action, next_state, reward)
                state = next_state
        for state in range(self.env.Ns):
            self.policy[state][np.argmax(self.Q[state])] = 1


class QLearning(Control):
    """
    Perform Q learning using epsilon greedy as a behaviour policy
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super(Control, self).__init__(env, n_episodes, alpha)
        self.epsilon = epsilon

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q, self.epsilon)

    def update(self, state, action, next_state, reward):
        next_action = np.argmax(self.Q[next_state])
        self.Q[state, action] += self.alpha * (
            reward
            + self.env.gamma * self.Q[next_state, next_action]
            - self.Q[state, action]
        )


class Sarsa(Control):
    """
    Sarsa with an epsilon greedy as a behaviour policy
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super(Control, self).__init__(env, n_episodes, alpha)
        self.epsilon = epsilon

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q, self.epsilon)

    def update(self, state, action, next_state, reward):
        next_action = self.behave(next_state)
        self.Q[state, action] += self.alpha * (
            reward
            + self.env.gamma * self.Q[next_state, next_action]
            - self.Q[state, action]
        )

class ExpectedSarsa(Control):
    """
    Expected Sarsa with an epsilon greedy as a behaviour policy and a greedy one as the target policy
    N.B: in this case Expected Sarsa is just Q learning
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super(Control, self).__init__(env, n_episodes, alpha)
        self.epsilon = epsilon

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q, self.epsilon)

    def target_policy(self):
        target_policy=np.zeros((self.env.Ns,self.env.Na))
        for state in range(self.env.Ns):
            target_policy[state][np.argmax(self.Q[state])] = 1
        return target_policy

    def update(self, state, action, next_state, reward):
        target_policy=self.target_policy()
        self.Q[state, action] += self.alpha * (
            reward
            + self.env.gamma * np.sum(target_policy[next_state]*self.Q[next_state])
            - self.Q[state, action]
        )
