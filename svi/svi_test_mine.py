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

from pyro.distributions.testing.fakes import NonreparameterizedNormal
import pyro.poutine

from envs.pointmass import PointMass
# from .tracegraph_elbo import TraceGraph_ELBO
from .distributions import FlexibleBernoulli

T = 20

def rl_model(env):
    # run simulation with random actions from the prior
    crew = 0
    env.reset()
    mx_pos = env.max_position * th.ones(1)
    env.state[:1] = pyro.sample("s_0", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[1:] = 0
    for i in range(T):
        a = pyro.sample("a_%d" % i, dist.Uniform(-100 * th.ones(2), 100 * th.ones(2)))
        # a = pyro.sample("a_%d" % i, dist.Normal(th.zeros(2), 10 * th.ones(2)))
        p_state = env.state
        _, r, _, _ = env.step(a.reshape(-1).detach().numpy())  # ignoring the `done` signal ...
        crew += r
        O_dist = FlexibleBernoulli(th.FloatTensor([r/T]).exp())
        pyro.sample("O_%d" % i, O_dist, obs=1)
    # O_dist = dist.Bernoulli(th.FloatTensor([crew/T]).exp())
    # pyro.sample("O_%d" % i, O_dist, obs=1)
    
    # TODO figure out: what should the return value be? I just used cumulative reward for now
    # print('reward in model: ', crew)
    return th.FloatTensor([crew])
    

def rl_linear_guide(env, render=False):
    # Define net params, just a linear model right now:
    # W = pyro.param("W", th.randn((3, 1)))
    # b = pyro.param("b", th.randn((1, 1)))
    pyro.module("policy", policy)
    pyro.module("critic", critic)

    # TODO: In section 3.2 he mentions that the distribution of transitions in q must be fixed to p
    # so I think it means we should run the simulation inside the guide too (I don't know how to do it otherwise)
    # run simulation with "guided" actions
    crew = 0
    env.reset()
    # TODO: find better way to sample states ...
    mx_pos = env.max_position * th.ones(1)
    env.state[:1] = pyro.sample("s_0", dist.Uniform(-1 * mx_pos, mx_pos))
    env.state[1:] = 0
    s = env.state
    for i in range(T):
        if render:
            env.render()
        a = pyro.sample(
            "a_%d" % i,
            NonreparameterizedNormal(
                # (th.FloatTensor(s).reshape(1,-1).mm(W) / env.max_position + b) * env.max_torque,
                policy(th.FloatTensor(s) / env.max_position) * env.max_torque,
                .5
            ),
            infer={'baseline': {
                'nn_baseline': critic,
                'nn_baseline_input': th.cat([th.FloatTensor(s), th.FloatTensor([i])]).detach()
            }},
        )
        s, r, _, _ = env.step(a.detach().reshape(-1).numpy())  # ignoring the `done` signal ...
        crew += r
    # print('reward in guide: ', crew)
    return th.FloatTensor([crew])



pyro.clear_param_store()
env = PointMass(reward_style='pot')   # the 'pot-mvel' reward is always negative

# Initializing the critic network (a.k.a. the value function)
critic = nn.Sequential(
    nn.Linear(4, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 2),
)

# policy network
policy = nn.Sequential(
    nn.Linear(3, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 8),
    nn.LeakyReLU(),
    nn.Linear(8, 1),
)

losses = []
optimizer = pyro.optim.Adam({"lr": 0.001})
loss_fn = pyro.infer.TraceGraph_ELBO(num_particles=20)

with pyro.poutine.trace(param_only=True) as param_capture:
        loss = loss_fn.loss_and_grads(rl_model, rl_linear_guide, env)
losses.append(loss)
params = set(site["value"].unconstrained()
        for site in param_capture.trace.nodes.values())
optimizer(params)
pyro.infer.util.zero_grads(params)
pre_guide_trace = pyro.poutine.trace(rl_linear_guide).get_trace(env)
# pre_guide_trace = loss_fn._get_traces(rl_model, rl_linear_guide, env)

import ipdb; ipdb.set_trace()
# for t in range(1, 10):
#     # step() takes a single gradient step and returns an estimate of the loss
#     # losses.append(svi.step(env))
#     # print('\rStep %d' % (t+1), end='')
#     with pyro.poutine.trace(param_only=True) as param_capture:
#         loss = loss_fn.loss_and_grads(rl_model, rl_linear_guide, env)
#     losses.append(loss)

#     params = set(site["value"].unconstrained()
#             for site in param_capture.trace.nodes.values())
#     optimizer(params)
#     pyro.infer.util.zero_grads(params)
#     # losses.append(loss)
#     # optimizer.step()
#     # optimizer.zero_grad()
#     pre_guide_trace = pyro.poutine.trace(rl_linear_guide).get_trace(env)
#     # model_trace = pyro.poutine.trace(rl_model).get_trace(env)
#     # import ipdb; ipdb.set_trace()
#     print('------Step %d' % (t+1))
#     print(guide_trace.nodes['a_10']['value'])
#     print(loss)

# W2 = pyro.param("W2")
# b2 = pyro.param("b2")
W = pyro.param("W")
b = pyro.param("b")
print('W:', W)
print('b:', b)
    
# print('W2:', W)
# print('b2:', b)

# Visualizing the learned policy
rl_linear_guide(env, True)
rl_linear_guide(env, True)
rl_linear_guide(env, True)
rl_linear_guide(env, True)
rl_linear_guide(env, True)
rl_linear_guide(env, True)

p, = plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
p.get_figure().canvas.draw()
p.get_figure().canvas.flush_events()
plt.show()
# time.sleep(100)
