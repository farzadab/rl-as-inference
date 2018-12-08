'''
Simplest possible test: finding a fixed direction with MSE

Report:
  - it works, but it does take a bit of time
    * depending on the starting point, takes around 200-500 steps with lr=0.01 and num_particles=100
  - lr=0.001: needs a lot more than 1000 steps
  - num_particles=10 and lr=0.001: solves after ~7K (more efficient than num_particles=100
  - num_particles=1  and lr=0.001: solves after ~20K (a lot more efficient!)
  - num_particles=1  and lr=0.01 : solves the problem but doesn't quite settle on the right solution (moves around)
  - num_particles=10 and lr=0.01 : same as above but with a less variation

  - num_particles=10 and lr=0.003: solves after ~2K
Remark:
  - lr * steps ~= 2-20 is required almost regardless of the num_particles
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

def rl_model():
    a = pyro.sample("a", dist.Uniform(-100 * th.ones(2), 100 * th.ones(2)))
    # reward is the distance to the correct direction
    r = -1 * (a - direction).pow(2).sum()
    O_dist = FlexibleBernoulli(th.FloatTensor([r]).exp())
    pyro.sample("O", O_dist, obs=1)
    return th.cat([a.detach(), th.FloatTensor([r])])
    

def rl_linear_guide():
    # Define net params, just a linear model right now:
    W = pyro.param("W", th.randn(2))

    a = pyro.sample(
        "a",
        NonreparameterizedNormal(
            W,
            .5
        ),
    )
    r = -1 * (a.detach() - direction).pow(2).sum()
    return th.cat([a.detach(), th.FloatTensor([r])])


pyro.clear_param_store()
svi = pyro.infer.SVI(model=rl_model,
                     guide=rl_linear_guide,
                     optim=pyro.optim.Adam({"lr": 0.003}),
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
time.sleep(100)