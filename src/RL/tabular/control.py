import numpy as np
from RL.tabular.utils import eps_greedy
from RL.envs.test_env import ToyEnv1
import itertools

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

    def sample_action(self,state,policy):
        return np.random.choice(np.arange(self.env.Na),p=policy[state])

    def compute_returns(self,rewards):
        gammas=np.array([self.env.gamma**i for i in range(len(rewards))])
        return np.sum(gammas*np.array(rewards))

    def update(self, state, action, next_state, reward):
        """
         Given the current state, current action, the reward and the next state, performs an online update in each step of an episode
         """
        raise NotImplementedError

    def update(self, trajectory):
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
                self.update(state, action, next_state, reward)
                state = next_state
        for state in range(self.env.Ns):
            self.policy[state][np.argmax(self.Q[state])] = 1

    def run_offline(self):
        for episode in range(self.n_episodes):
            trajectory={'states':[],'actions':[],'rewards':[]}
            done = False
            state = self.env.reset()
            while not done:
                action = self.sample_action(state,self.policy)
                next_state, reward, done, info = self.env.step(action)
                trajectory['actions'].append(action); trajectory['states'].append(state); trajectory['rewards'].append(reward)
                state = next_state
            self.update(trajectory)
            self.policy*=0
            for state in range(self.env.Ns):
                self.policy[state][np.argmax(self.Q[state])] = 1




class QLearning(Control):
    """
    Perform Q learning using epsilon greedy as a behaviour policy
    """

    def __init__(self, env, n_episodes=10, alpha=0.01, epsilon=0.1):
        super(QLearning, self).__init__( env, n_episodes, alpha)
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
        super(Sarsa, self).__init__( env, n_episodes, alpha)
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
        super(ExpectedSarsa, self).__init__( env, n_episodes, alpha)
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

class MonteCarlo(Control):
    def __init__(self, env, n_episodes=10):
        super(MonteCarlo,self).__init__(env,n_episodes)
        self.returns=[[[] for a in range(self.env.Na)] for s in range(self.env.Ns)]
        self.policy=np.random.rand(self.env.Ns,self.env.Na)
        row_sums = self.policy.sum(axis=1)
        self.policy = self.policy / row_sums[:, np.newaxis] #is now a valid probability
        self.pairs = list(itertools.product(np.arange(self.env.Ns), np.arange(self.env.Ns)))


    def update(self, trajectory):
        total_pairs=len(self.pairs)
        done=[]
        i=0
        while len(done)<total_pairs and i<len(trajectory['rewards']):
            state,action=trajectory['states'][i],trajectory['actions'][i]
            if (state,action) not in done:
                self.returns[state][action].append(self.compute_returns(trajectory['rewards'][i:]))
                done.append((state,action))
                self.Q[state][action]=np.mean(self.returns[state][action])
            i+=1







if __name__ == "__main__":
    env = ToyEnv1(gamma=0.99)
    n_episodes = 1000
    alpha = 0.1
    epsilon = 0.1
    algo=MonteCarlo(env,n_episodes)
    algo.run_offline()

    print(algo.policy)


