'''
2nd simple test: state-dependent direction with MSE reward

Report:
  - num_particles=10  and lr=0.003: solves after ~2-3K
  - num_particles=100 and lr=0.003: solves after ~2-3K (really inefficient)
  - num_particles=1   and lr=0.003: solves after ~8-9K (more efficient, but noisy ELBO)
    * can't really tell from ELBO alone when/if the solution has been found
  - num_particles=10  and lr=0.001: solves after ~7-8K (in line with the lr * steps formula from previous experiments)
  - num_particles=10  and lr=0.01 : solves after ~1-2K but doesn't settle down even after 5K
'''
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

from envs.pointmass import PointMass
from .tracegraph_elbo import TraceGraph_ELBO
from .distributions import FlexibleBernoulli


direction = th.FloatTensor([1, -1])

def init_state():
    mx_pos = th.ones(2)
    p = pyro.sample("s", dist.Uniform(-1 * mx_pos, mx_pos))
    g = pyro.sample("g", dist.Uniform(-1 * mx_pos, mx_pos))
    return p, g

def compute_reward(a, p, g):
    return -1 * (a - (g - p)).pow(2).sum()

def rl_model():
    p, g = init_state()
    a = pyro.sample("a", dist.Uniform(-100 * th.ones(2), 100 * th.ones(2)))
    # reward is the distance to the correct direction
    r = compute_reward(a, p, g)
    O_dist = FlexibleBernoulli(th.FloatTensor([r]).exp())
    pyro.sample("O", O_dist, obs=1)
    return th.cat([a.squeeze().detach(), th.FloatTensor([r])])
    

def rl_linear_guide():
    # Define net params, just a linear model right now:
    W = pyro.param("W", th.randn((4, 2)))

    p, g = init_state()

    a = pyro.sample(
        "a",
        NonreparameterizedNormal(
            th.cat([p, g]).reshape(1, -1).mm(W),
            .5
        ),
    )
    r = compute_reward(a, p, g)
    return th.cat([a.squeeze().detach(), th.FloatTensor([r])])


pyro.clear_param_store()
svi = pyro.infer.SVI(model=rl_model,
                     guide=rl_linear_guide,
                     optim=pyro.optim.Adam({"lr": 0.01}),
                     loss=pyro.infer.Trace_ELBO(num_particles=10))

losses = []
for t in range(10000):
    # step() takes a single gradient step and returns an estimate of the loss
    losses.append(svi.step())
    if t % 100 == 0:
        print('Step %d' % (t+1), end='')
        print('  W:', pyro.param("W"))


p, = plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
p.get_figure().canvas.draw()
p.get_figure().canvas.flush_events()
plt.show()
time.sleep(10000)