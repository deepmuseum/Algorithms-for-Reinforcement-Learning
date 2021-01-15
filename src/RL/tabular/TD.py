import numpy as np
from RL.tabular.utils import eps_greedy, Control, Prediction
from RL.envs.test_env import ToyEnv1
from RL.envs.cliffwalk import CliffWalk


class QLearning(Control):
    """
    Perform Q learning using epsilon greedy as a behaviour policy
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super(QLearning, self).__init__(env, n_episodes, alpha)
        self.epsilon = epsilon

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q, self.epsilon)

    def update_online(self, state, action, next_state, reward):
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
        super(Sarsa, self).__init__(env, n_episodes, alpha)
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
        super().__init__(env, n_episodes, alpha)
        self.epsilon = epsilon

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q, self.epsilon)

    def target_policy(self):
        target_policy = np.zeros((self.env.Ns, self.env.Na))
        for state in range(self.env.Ns):
            target_policy[state][np.argmax(self.Q[state])] = 1
        return target_policy

    def update(self, state, action, next_state, reward):
        target_policy = self.target_policy()
        self.Q[state, action] += self.alpha * (
            reward
            + self.env.gamma * np.sum(target_policy[next_state] * self.Q[next_state])
            - self.Q[state, action]
        )


class DoubleQLearning(Control):
    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super().__init__(env, n_episodes, alpha)
        self.epsilon = epsilon
        self.Q_ = np.zeros((env.Ns, env.Na))

    def behave(self, state):
        return eps_greedy(state, self.env.Na, self.Q + self.Q_, self.epsilon)

    def update_online(self, state, action, next_state, reward):
        if np.random.uniform() < 0.5:
            next_action = np.argmax(self.Q[next_state])
            self.Q[state, action] += self.alpha * (
                reward
                + self.env.gamma * self.Q_[next_state, next_action]
                - self.Q[state, action]
            )
        else:
            next_action = np.argmax(self.Q_[next_state])
            self.Q_[state, action] += self.alpha * (
                reward
                + self.env.gamma * self.Q[next_state, next_action]
                - self.Q_[state, action]
            )


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


if __name__ == "__main__":

    # env = CliffWalk()

    env = ToyEnv1(gamma=0.99)
    n_episodes = 100
    alpha = 0.1
    epsilon = 0.1
    algo = DoubleQLearning(env, n_episodes, alpha, epsilon)
    algo.run_online()

    print(algo.policy)
