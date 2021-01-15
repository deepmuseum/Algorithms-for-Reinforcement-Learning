import itertools
import numpy as np
from RL.tabular.utils import eps_greedy, Control, Prediction
from RL.envs.test_env import ToyEnv1
from RL.envs.cliffwalk import CliffWalk


class MonteCarlo(Control):
    def __init__(self, env, n_episodes=10):
        super().__init__(env, n_episodes)
        self.returns = [[[] for _ in range(self.env.Na)] for _ in range(self.env.Ns)]
        self.policy = np.random.rand(self.env.Ns, self.env.Na)
        row_sums = self.policy.sum(axis=1)
        self.policy = (
            self.policy / row_sums[:, np.newaxis]
        )  # is now a valid probability
        self.pairs = list(
            itertools.product(np.arange(self.env.Ns), np.arange(self.env.Ns))
        )

    def update_offline(self, trajectory):
        total_pairs = len(self.pairs)
        done = set()
        i = 0
        while len(done) < total_pairs and i < len(trajectory["rewards"]):
            state, action = trajectory["states"][i], trajectory["actions"][i]
            if (state, action) not in done:
                self.returns[state][action].append(
                    self.compute_returns(trajectory["rewards"][i:])
                )
                done.add((state, action))
                self.Q[state][action] = np.mean(self.returns[state][action])
            i += 1


if __name__ == "__main__":
    # env = CliffWalk()

    env = ToyEnv1(gamma=0.99)
    n_episodes = 100
    alpha = 0.1
    epsilon = 0.1
    algo = MonteCarlo(env, n_episodes)
    algo.run_offline()

    print(algo.policy)
