import numpy as np
from RL.tabular.utils import eps_greedy, Control, Prediction, Nstep, proba_eps_greedy
from RL.envs.test_env import ToyEnv1
from RL.envs.cliffwalk import CliffWalk


class TD(Nstep):
    """
    Perform n-step TD for estimating V
    """

    def __init__(self, env, policy, n_episodes=10, alpha=0.01, n=1):
        super(TD, self).__init__(env, policy, n_episodes, alpha, n)

    def act(self, done, next_state, t):
        if done:
            return 0, t + 1
        else:
            next_action = self.sample_action(next_state, self.policy)
            return next_action, float("inf")

    def update(self, trajectory, to, T):
        h = min(to + self.n, T)
        gammas = np.power(self.env.gamma, np.arange(h - to - 1))
        G = np.sum(gammas * np.array(trajectory["rewards"][to + 1 : h + 1]))
        if to + self.n < T:
            G += self.env.gamma ** self.n * self.V[trajectory["states"][to + n]]
        self.V[trajectory["states"][to]] = self.V[
            trajectory["states"][to]
        ] + self.alpha * (G - self.V[trajectory["states"][to]])


class Sarsa(Nstep):
    """
    Perform n-step Sarsa for estimating Q_pi or Q*
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, n=1, policy=None, epsilon=0.1):
        super(Sarsa, self).__init__(env, n_episodes, alpha, n, policy, epsilon)

    def act(self, done, next_state, t):
        if done:
            return 0, t + 1
        else:
            if self.problem == "control":
                next_action = eps_greedy(next_state, self.env.Na, self.Q, self.epsilon)
            else:
                next_action = self.sample_action(next_state, self.policy)
            return next_action, float("inf")

    def update(self, trajectory, to, T):
        h = min(to + self.n, T)
        gammas = np.power(self.env.gamma, np.arange(h - to - 1))
        G = np.sum(gammas * np.array(trajectory["rewards"][to + 1 : h + 1]))
        if to + self.n < T:
            G += (
                self.env.gamma ** self.n
                * self.Q[trajectory["states"][to + n], trajectory["actions"][to + n]]
            )
        self.Q[trajectory["states"][to], trajectory["actions"][to]] = self.Q[
            trajectory["states"][to], trajectory["actions"][to]
        ] + self.alpha * (
            G - self.Q[trajectory["states"][to], trajectory["actions"][to]]
        )


class SarsaOff(Nstep):
    """
    Perform off policy n-step Sarsa for estimating Q_pi or Q*
    """

    def __init__(
        self,
        env,
        n_episodes=10,
        alpha=0.01,
        n=1,
        policy=None,
        epsilon=0.1,
        behavior=None,
    ):
        super(SarsaOff, self).__init__(env, n_episodes, alpha, n, policy, epsilon)
        self.behavior = behavior

    def act(self, done, next_state, t):
        if done:
            return 0, t + 1
        else:
            next_action = self.sample_action(next_state, self.behavior)
            return next_action, float("inf")

    def update(self, trajectory, to, T):
        h = min(to + self.n, T)
        proba = proba_eps_greedy(self.env.Ns, self.env.Na, self.Q, self.epsilon)
        gho = np.product(
            np.array(
                [
                    proba[trajectory["states"][i], trajectory["actions"][i]]
                    for i in range(to + 1, h + 1)
                ]
            )
            * np.array(
                [
                    self.behavior[trajectory["states"][i], trajectory["actions"][i]]
                    for i in range(to + 1, h + 1)
                ]
            )
        )
        gammas = np.power(self.env.gamma, np.arange(h - to - 1))
        G = np.sum(gammas * np.array(trajectory["rewards"][to + 1 : h + 1]))
        if to + self.n < T:
            G += (
                self.env.gamma ** self.n
                * self.Q[trajectory["states"][to + n], trajectory["actions"][to + n]]
            )
        self.Q[trajectory["states"][to], trajectory["actions"][to]] = self.Q[
            trajectory["states"][to], trajectory["actions"][to]
        ] + gho * self.alpha * (
            G - self.Q[trajectory["states"][to], trajectory["actions"][to]]
        )


if __name__ == "__main__":

    # env = CliffWalk()
    env = ToyEnv1(gamma=0.99)
    policy = np.random.rand(env.Ns, env.Na)
    row_sums = policy.sum(axis=1)
    policy = policy / row_sums[:, np.newaxis]

    n_episodes = 100
    alpha = 0.1
    n = 2
    algo = SarsaOff(env, n_episodes, alpha, n, None, 0.1, policy)
    algo.run()

    print(algo.Q)
