> figures are taken from Sutton's RL book

# Policy gradient algorithms

Algorithms based on the policy gradient theorem :

![Policy gradient theorem](../../../../figures/policy_gradient_theorem.png)


## Reinforce :

![reinforce pseudo code](../../../../figures/reinforce_pseudo_code.png)

## Reinforce with baseline :


## Advantage Actor-Critic (A2C) :

In actor-critic methods, we use bootstrapping (i.e. updating the estimated
value of the current state with the estimated values of other states)
This technique allows reducing variance of estimates and accelerates learning.

One step actor-critic replaces the full return of REINFORCE with learned value function
baseline by the one-step return :

![Actor-Critic update](../../../../figures/actor_critic_pseudo_code.png)
