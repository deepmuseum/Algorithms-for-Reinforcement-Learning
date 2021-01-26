import torch
from torch import nn


class SarsaLambda:
    def __init__(self, env, lambda_factor, q_network):
        self.env = env
        self.lambda_factor = lambda_factor
        self.q_network = q_network
        self.eligibility_trace = 0


if __name__ == "__main__":
    pass
