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
