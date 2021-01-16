import numpy as np
from RL.tabular.utils import eps_greedy, Control, Prediction, PredictionNstep
from RL.envs.test_env import ToyEnv1
from RL.envs.cliffwalk import CliffWalk


class TDprediction(PredictionNstep):
    """
    Perform n-step TD for estimating V
    """

    def __init__(self, env, policy, n_episodes=10, alpha=0.01, n=1):
        super(TDprediction, self).__init__(env, policy, n_episodes, alpha, n)

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


if __name__ == "__main__":

    # env = CliffWalk()
    env = ToyEnv1(gamma=0.99)
    policy = np.random.rand(env.Ns, env.Na)
    row_sums = policy.sum(axis=1)
    policy = policy / row_sums[:, np.newaxis]

    n_episodes = 100
    alpha = 0.1
    n = 2
    algo = TDprediction(env, policy, n_episodes, alpha, n)
    algo.run()

    print(algo.V)
