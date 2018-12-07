import numpy as np
import matplotlib.pyplot as plt
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from envs.pointmass import PointMass

from vectorized_policy_gradient import *

T = 50

def rl_model(env, last_model=None):
    # run simulation with random actions from the prior
    crew = 0
    state = env.reset()
    mx_pos = env.max_position * th.ones(2)
    env.state[:2] = pyro.sample("s_0", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[2:] = 0
    for i in range(T):

        # check how we are sampling actions from their prior
        if last_model == None:
            # no beliefs about optimal action then a wide normal
            a = pyro.sample("a_%d" % i, dist.Normal(th.zeros(2), 10 * th.ones(2)))
        else:
            # if we have an old model, sample from that (assumed to be dist.Normal(state*W+b, Sigma) though)
            a = pyro.sample("a_%d" % i, last_model(state))

        state, r, _, _ = env.step(a.reshape(-1).detach().numpy())  # ignoring the `done` signal ...
        crew += r
        O_dist = dist.Bernoulli(th.FloatTensor([r]).exp())
        pyro.sample("O_%d" % i, O_dist, obs=1)

    # TODO figure out: what should the return value be? I just used cumulative reward for now
    return th.FloatTensor([crew])


def rl_linear_guide(env, critic, render=False):
    # Define net params, just a linear model right now:
    W = pyro.param("W", th.randn((6, 2)))
    b = pyro.param("b", th.randn((1, 2)))

    # TODO: In section 3.2 he mentions that the distribution of transitions in q must be fixed to p
    # so I think it means we should run the simulation inside the guide too (I don't know how to do it otherwise)

    # run simulation with "guided" actions
    crew = 0
    env.reset()
    # TODO: find better way to sample states ...
    mx_pos = env.max_position * th.ones(2)
    env.state[:2] = pyro.sample("s_0", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[2:] = 0
    s = env.state
    for i in range(T):
        if render:
            env.render()
        a = pyro.sample(
            "a_%d" % i,
            dist.Normal(
                th.FloatTensor(s).reshape(1,-1).mm(W) + b,
                .2
            ),
            infer={'baseline': {
                'nn_baseline': critic,
                'nn_baseline_input': s
            }},
        )
        s, r, _, _ = env.step(a.detach().reshape(-1).numpy())  # ignoring the `done` signal ...
        crew += r
    return th.FloatTensor([crew])

def train_regression(iterations = 100, old_model = None):

    pyro.clear_param_store()

    # Initializing the critic network (a.k.a. the value function)
    critic = nn.Sequential(
        nn.Linear(6, 4),
        nn.LeakyReLU(),
        nn.Linear(4, 4),
        nn.LeakyReLU(),
        nn.Linear(4, 2),
        nn.LeakyReLU(),
        nn.Linear(2, 1),
    )

    if old_model == None:
        rl_linear_guide_ = lambda env: rl_linear_guide(env, critic)
        svi = pyro.infer.SVI(model=rl_model,
                             guide=rl_linear_guide_,
                             optim=pyro.optim.Adam({"lr": 0.001}),
                             loss=pyro.infer.Trace_ELBO(num_particles=20))
    else:
        rl_linear_guide_ = lambda env: rl_linear_guide(env, critic)
        rl_model_ = lambda env: rl_model(env, old_model)
        svi = pyro.infer.SVI(model=rl_model_,
                             guide=rl_linear_guide_,
                             optim=pyro.optim.Adam({"lr": 0.001}),
                             loss=pyro.infer.Trace_ELBO(num_particles=20))

    env = PointMass(reward_style='distsq')   # the 'distsq' reward is always negative

    losses = []
    for t in range(iterations):
        # step() takes a single gradient step and returns an estimate of the loss
        losses.append(svi.step(env))
        print('\rStep %d' % (t+1), end='')

    W = pyro.param("W")
    b = pyro.param("b")
    # W2 = pyro.param("W2")
    # b2 = pyro.param("b2")

    print('W:', W)
    print('b:', b)
    # print('W2:', W)
    # print('b2:', b)

    # # Visualizing the learned policy
    # rl_linear_guide(env, critic, True)
    # rl_linear_guide(env, critic, True)
    # rl_linear_guide(env, critic, True)
    #
    # # p, = plt.plot(losses)
    # # plt.title("ELBO")
    # # plt.xlabel("step")
    # # plt.ylabel("loss")
    # # p.get_figure().canvas.draw()
    # # p.get_figure().canvas.flush_events()
    # # plt.show()
    # # time.sleep(100)

    return W, b
# now we have a guess of the model, lets try it using the old waits

def main():

    # get an initial set of values
    W, b = train_regression(100)
    # now lets try re-using our old guess as the new proir on actions
    for iter in range(1,5):
        # still use wide variance to start though
        old_model = lambda state: dist.Normal(
            th.FloatTensor(state).reshape(1,-1).mm(W) + b, 10/(iter**2)*th.ones(2))
        W, b = train_regression(100, old_model)

if __name__ == '__main__':
    main()
